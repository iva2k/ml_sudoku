# RL Sudoku

## 🧠 Low-Level RL Concepts

The Sudoku problem can be framed as a **Markov Decision Process (MDP)**, the formal structure for RL problems.

---

### 1. The Markov Decision Process (MDP)

* **Agent:** The solver (neural network).
* **Environment:** The Sudoku grid and the rules of the game.
* **State ($S$):** The current configuration of the $9 \times 9$ Sudoku board. This can be represented as a $9 \times 9$ matrix where empty cells are $0$ and filled cells are $1-9$.
* **Action ($A$):** The set of all possible moves the agent can make. A move is defined by placing a digit ($1-9$) into a specific empty cell (row, col). The total action space size is $9 \times 9 \times 9 = 729$.
* **Reward ($R$):** The scalar feedback the environment gives the agent after an action. This is the crucial part to design:
  * **Positive Reward:** $\uparrow$ for making a valid move that respects all Sudoku rules (row, column, $3 \times 3$ box). A large $\uparrow$ for solving the entire puzzle.
  * **Negative Reward (Penalty):** $\downarrow$ for making an invalid move (e.g., placing a number that violates a rule or trying to fill an already filled cell). A small $\downarrow$ for every step to encourage efficiency.
* **Policy ($\pi$):** The agent's strategy. It maps a given state $S$ to a probability distribution over actions $A$, i.e., $\pi(a|s)$. The goal is to learn the optimal policy $\pi^*$.
* **Reward Design: Pure RL vs. Supervised Guidance:**
  * **Pure RL:** A pure RL approach would only reward the agent for following the *rules* of Sudoku, not for knowing the *answer*. The agent would have to discover the correct digit for a cell through trial and error. While theoretically sound, the action space is so vast that the agent often gets stuck in a loop of making invalid moves and never learns the underlying logic.
  * **Supervised Guidance (Practical Approach):** To make training feasible, we can introduce a supervised element. Instead of rewarding based on rule validity, we reward the agent for placing a digit that matches the pre-computed solution. This provides a much stronger and more direct learning signal, effectively teaching the agent to "imitate" the solution. While this is less of a pure discovery process, it's a practical compromise to ensure the agent learns effectively. Our implementation uses this supervised approach, but it can be turned off.
* **Value Function ($Q(s, a)$):** The expected cumulative discounted reward starting from state $s$ and taking action $a$, then following policy $\pi$ thereafter. A neural network will often be used to approximate this function.

---

### 2. Deep Q-Network (DQN) Algorithm

Since the state space (all possible Sudoku grids) is too large to store a traditional Q-table, we'll use a deep neural network to approximate the Q-function, leading to **Deep Q-Learning (DQN)**.

* **Q-Network:** A neural network $\theta$ (a CNN/DNN) that takes the state $S$ as input and outputs the Q-values for all $729$ possible actions. $Q(S, A; \theta)$.
* **Target Network:** A second, identical network $\theta^{-}$ whose parameters are updated less frequently (e.g., every $N$ steps) from the main Q-Network. This stabilizes the training.
* **Bellman Equation & Loss:** The network is trained by minimizing the **Mean Squared Error (MSE) loss** between the current Q-value and the target Q-value, which is derived from the Bellman equation:
$$L(\theta) = E_{s, a, r, s'} \left[ \left( Q(s, a; \theta) - y \right)^2 \right]$$
Where the target $y$ is:
$$y = r + \gamma \max_{a'} Q(s', a'; \theta^{-})$$
* **Experience Replay:** A buffer (deque) is used to store past experiences $(s, a, r, s', \text{done})$. Training samples are drawn randomly from this buffer, which breaks the correlation between sequential states and greatly stabilizes learning.

---

## 🏗️ High-Level Implementation Steps

### 1. Environment Implementation

The environment defines the Sudoku game logic. We will use the `gymnasium` (formerly `gym`) library standard for a robust setup.

* **State Representation:**

  * Input to the network: A $9 \times 9 \times 1$ tensor (for a CNN) or a flattened vector of $81$ elements (for a DNN). Values are $0-9$.
  * **Alternative Input:** To make the CNN learn more easily, we could use a **one-hot encoding** of the board, resulting in a $9 \times 9 \times 10$ input tensor, where the last dimension represents $0$ (empty) and $1-9$. This is often superior for $\text{CNN}$s.
* **Action Mapping:** Map the network's $729$-dimensional output (Q-values) back to a $(row, col, digit)$ triplet for the move.
* **Transition ($\text{step}$ function):** This function takes an action, updates the state, and returns:
  1. The **next state** $S'$.
  2. The **reward** $R$.
  3. A **done** flag (True if the puzzle is solved or an irreversible mistake is made).
  4. An **info** dictionary (e.g., for debugging metrics).

### 2. Network Model ($\text{Q-Network}$)

A **CNN-based Q-Network** is an excellent choice as it can naturally capture the local structure (rows, columns, $3 \times 3$ boxes) of the Sudoku board.

* **Input Layer:** $9 \times 9 \times (\text{features})$, e.g., $9 \times 9 \times 10$ if using one-hot encoding.
* **Convolutional Layers:** Use several $\text{Conv2D}$ layers (e.g., $3 \times 3$ filters, $\text{ReLU}$ activation) to extract features about valid/invalid moves across the grid. The filters will learn to detect Sudoku rule violations.
* **Flatten and $\text{Dense}$ Layers:** Flatten the output of the CNN and pass it through a few $\text{Dense}$ layers to aggregate the features.
* **Output Layer:** The final $\text{Dense}$ layer must have **$729$ outputs**, one for each possible action $(r, c, d)$. No activation (or a linear one) is used since the output is a Q-value (a score).

### 3. Training Setup and Hyperparameters

* **RL Algorithm:** **Deep Q-Learning (DQN)** is a great starting point for this discrete action space.
* **Policy:** Use an **$\epsilon$-greedy policy** for action selection:
  * With probability $\epsilon$ (high initially, decaying over time), choose a **random** action (Exploration).
  * With probability $1 - \epsilon$, choose the **greedy** action (Exploitation) with the highest Q-value: $a = \operatorname{argmax}_a Q(s, a; \theta)$.
* **Hyperparameters:**
  * **Discount Factor ($\gamma$):** Close to $1$ (e.g., $0.99$). Since solving Sudoku is an episodic task, a high $\gamma$ is needed so the agent values the final, large reward for solving the puzzle.
  * **Learning Rate ($\alpha$):** A small value (e.g., $10^{-4}$ to $10^{-3}$) for the optimizer ($\text{Adam}$ is standard).
  * **Experience Replay Size:** Large (e.g., $10,000$ to $100,000$) to store many episodes.
  * **Batch Size:** Standard for deep learning (e.g., $32$ to $128$).
  * **Target Network Update Frequency:** Update $\theta^{-} \leftarrow \theta$ every $C$ steps (e.g., $C=1000$).

## Three-Phase Implementation Plan

### Phase 1: Quick Setup and Code Carcass (Goal: Runnable Structure)

This phase establishes the foundational structure, including imports, command-line arguments, and the skeletal definition of the three main components: the Environment, the Network, and the Replay Buffer.

**Steps:**

1. **Setup Environment:** Define the necessary libraries in a virtual environment (`gymnasium`, `torch`, `numpy`).
2. **Define Carcass:** Create the primary Python file (`sudoku_rl.py`) containing the `main`, argument parsing, and placeholder classes for `SudokuEnv`, `DQNSolver`, and `ReplayBuffer`.
3. **Basic $\text{main}$ Function:** Implement the core loop of initialization (env, agent, network) and a basic episode loop to test environment interaction (even if `step` returns dummy values).

### Phase 2: Core Logic Implementation (Goal: Functioning Environment)

This phase focuses on the heart of the Sudoku problem: defining the rules and state transitions.

**Steps:**

1. **Environment Initialization (`__init__` and `reset`):** Implement puzzle loading and state representation (a $9 \times 9$ array/tensor). A simple starting puzzle is sufficient initially.
2. **Action Validation and Reward:** Implement the `SudokuEnv.step` method. This is critical.
   * Parse the action $(r, c, d)$.
   * Check if the cell $(r, c)$ is empty.
   * Check if placing $d$ violates Sudoku rules (row, col, $3 \times 3$ box).
   * Assign a simple reward: large positive reward for a *valid* move, large negative penalty for an *invalid* move (violating rules), small negative penalty for trying to overwrite a fixed cell.
3. **Done Condition:** Check if the grid is fully filled *and* valid (solved).

### Phase 3: Deep Q-Learning (DQN) Training (Goal: Learning Agent)

This phase connects the environment to the PyTorch neural network and implements the learning algorithm.

**Steps:**

1. **Network Implementation (`DQNSolver`):** Build the CNN model. Given the $9 \times 9$ grid, $\text{Conv2D}$ layers are ideal for spatial reasoning. The output layer must have $729$ dimensions (one for every possible action).
2. **Replay Buffer Logic:** Implement `push` (store experience tuple) and `sample` (retrieve a random batch) for the `ReplayBuffer`.
3. **Training Function (`optimize_model`):** Implement the DQN core: sample a batch, calculate the target Q-value using the Bellman equation and the **Target Network**, calculate MSE loss, and run backpropagation.
4. **Full Training Loop:** Integrate $\epsilon$-greedy policy selection, step-by-step interaction with the environment, storage in the buffer, and periodic calls to `optimize_model`.

## Improvements

### Action masking

That's a very common and important issue when applying Reinforcement Learning to constrained environments like Sudoku. What we can observe is the agent spending most of its time exploring the vast number of **illegal actions** (e.g., trying to place a number in a fixed starting cell, or trying to place a number in an already filled cell).

The current action space has $9 \times 9 \times 9 = 729$ total actions. If only 10 spots are empty, $719$ of those actions are pointless. The agent must waste thousands of steps to learn that these $719$ actions always lead to a harsh penalty.

The best solution is **Action Masking** (or **Action Filtering**). This forces the agent's policy to only consider actions that are *legal* (i.e., placing a number in a non-fixed, currently empty cell).

![Action Masking in Reinforcement Learning](images/agent-environment.jpg)

We will introduce the `--masking` parameter. When enabled, the agent will only select actions that target empty, non-fixed cells, dramatically speeding up exploration and focusing learning on the true Sudoku rules.

#### Key Changes in `rl_sudoku.py`

1. **`generate_legal_mask(grid)`:** A new helper function that generates a boolean mask (size 729) where `True` indicates the action targets an empty cell.
2. **`get_action`:**
   * If masking is enabled, the agent samples a random action only from the set of **legal actions** during exploration ($\epsilon$-greedy).
   * During exploitation (greedy action), the Q-values for illegal actions are set to a very large negative number (`-1e10`), ensuring the agent never selects them.
3. **`optimize_model`:** The same masking logic is applied to the **Target Network's** Q-values when calculating the optimal future value $V(s')$, which is essential for stable learning.
4. **`main`:** A new command-line argument `--masking` has been added.

### Curriculum Learning: Staged Difficulty

Starting the training with very difficult Sudoku puzzles (e.g., only 25 clues) can be overwhelming for the agent. It's like asking a new student to solve an expert-level problem. The agent may fail to learn meaningful patterns.

A better approach is **Curriculum Learning**, where the agent is first trained on easier problems and the difficulty is gradually increased as its performance improves.

#### Implementation Plan

We will implement a difficulty "staircase" based on the training episode number.

1. **Initial Phase (Easy):** For the first block of episodes, the environment will generate puzzles with a high number of clues (e.g., 70-80). This makes it easier for the agent to find correct moves and learn the basic structure.
2. **Intermediate Phase (Medium):** As training progresses, we reduce the number of clues (e.g., 50-75). The agent must now learn to solve puzzles requiring more steps and more complex deductions.
3. **Final Phase (Hard):** In the later stages of training, the number of clues is reduced to a standard difficult range (e.g., 25-55), forcing the agent to generalize its learned policy to harder problems.

This staged approach helps stabilize training and leads to a more robust final policy.
