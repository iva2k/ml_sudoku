# Sudoku RL Agent

This project implements a Reinforcement Learning agent to solve Sudoku puzzles.

It uses a Deep Q-Network (DQN) with various architectural enhancements and training strategies like action masking, reward shaping, and curriculum learning.

## 🧠 Low-Level RL Concepts

The Sudoku problem can be framed as a **Markov Decision Process (MDP)**, the formal structure for RL problems.

---

### 1. The Markov Decision Process (MDP)

* **Agent:** The solver (neural network).
* **Environment:** The Sudoku grid and the rules of the game.
* **State ($S$):** The current configuration of the $9 \times 9$ Sudoku board. This can be represented as a $9 \times 9$ matrix where blanks (empty cells) are $0$ and filled cells are $1-9$.
* **Action ($A$):** The set of all possible moves the agent can make. A move is defined by placing a digit ($1-9$) into a specific blank (row, col). The total action space size is $9 \times 9 \times 9 = 729$.
* **Reward ($R$):** The scalar feedback the environment gives the agent after an action. This is the crucial part to design:
  * **Positive Reward:** $\uparrow$ for making a valid move that respects all Sudoku rules (row, column, $3 \times 3$ box). A large $\uparrow$ for solving the entire puzzle.
  * **Negative Reward (Penalty):** $\downarrow$ for making an invalid move (e.g., placing a number that violates a rule or trying to fill an already filled cell). A small $\downarrow$ for every step to encourage efficiency.
* **Policy ($\pi$):** The agent's strategy. It maps a given state $S$ to a probability distribution over actions $A$, i.e., $\pi(a|s)$. The goal is to learn the optimal policy $\pi^*$.
* **Reward Design: Pure RL vs. Supervised Guidance:**
  * **Pure RL:** A pure RL approach would only reward the agent for following the *rules* of Sudoku, not for knowing the *answer*. The agent would have to discover the correct digit for a cell through trial and error. While theoretically sound, the action space is so vast that the agent often gets stuck in a loop of making invalid moves and never learns the underlying logic.
  * **Hybrid Approach (Implemented):** To make training feasible and robust, our implementation uses a hybrid reward system that combines supervised guidance with rule-based rewards. This provides a strong learning signal while still encouraging valid exploration.
    * **`+100.0`:** For solving the puzzle completely.
    * **`+10.0`:** For placing a digit that matches the pre-computed solution.
    * **`-5.0`:** Penalty for an invalid move that violates Sudoku rules or deviates from the unique solution path.
    * **`-10.0`:** Penalty for trying to overwrite an already filled cell (though action masking largely prevents this).
* **Value Function ($Q(s, a)$):** The expected cumulative discounted reward starting from state $s$ and taking action $a$, then following policy $\pi$ thereafter. A neural network will often be used to approximate this function.

---

### 2. Deep Q-Network (DQN) Algorithm

Since the state space (all possible Sudoku grids) is too large to store in a traditional Q-table, we'll use a deep neural network to approximate the Q-function, leading to **Deep Q-Learning (DQN)**.

* **Q-Network:** A neural network $\theta$ (a CNN/DNN) that takes the state $S$ as input and outputs the Q-values for all $729$ possible actions. $Q(S, A; \theta)$.
* **Target Network:** A second, identical network $\theta^{-}$ whose parameters are updated less frequently (e.g., every $N$ steps) from the main Q-Network. This stabilizes the training.
* **Bellman Equation & Loss:** The network is trained by minimizing the **Smooth L1 Loss (Huber Loss)** between the current Q-value and the target Q-value. This loss function is more robust to outliers than Mean Squared Error (MSE), which helps stabilize training when large rewards and penalties are present. The loss is calculated based on the Bellman equation:
$$L(\theta) = E_{s, a, r, s'} \left[ \text{SmoothL1} \left( Q(s, a; \theta), y \right) \right]$$
Where the target $y$ is:
$$y = r + \gamma \max_{a'} Q(s', a'; \theta^{-})$$
* **Experience Replay:** A buffer (deque) is used to store past experiences $(s, a, r, s', \text{done})$. Training samples are drawn randomly from this buffer, which breaks the correlation between sequential states and greatly stabilizes learning.

---

## 🏗️ High-Level Implementation Steps

### 1. Environment Implementation

The environment defines the Sudoku game logic. We will use the `gymnasium` (formerly `gym`) library standard for a robust setup.

* **State Representation:**

  * Input to the network: A $9 \times 9 \times 1$ tensor (for a CNN) or a flattened vector of $81$ elements (for a DNN). Values are $0-9$.
  * **Alternative Input:** To make the CNN learn more easily, we could use a **one-hot encoding** of the board, resulting in a $9 \times 9 \times 10$ input tensor, where the last dimension represents $0$ (blank) and $1-9$. This is often superior for $\text{CNN}$s.
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
   * Check if the cell $(r, c)$ is blank.
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

The current action space has $9 \times 9 \times 9 = 729$ total actions. If only 10 spots are blank, $719$ of those actions are pointless. The agent must waste thousands of steps to learn that these $719$ actions always lead to a harsh penalty.

The best solution is **Action Masking** (or **Action Filtering**). This forces the agent's policy to only consider actions that are *legal* (i.e., placing a number in a non-fixed, currently blank cell).

![Action Masking in Reinforcement Learning](images/agent-environment.jpg)

We will introduce the `--masking` parameter. When enabled, the agent will only select actions that target blank, non-fixed cells, dramatically speeding up exploration and focusing learning on the true Sudoku rules.

#### Key Changes in `rl_sudoku.py`

1. **`generate_legal_mask(grid)`:** A new helper function that generates a boolean mask (size 729) where `True` indicates the action targets a blank cell.
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

### Early Termination on Invalid Solution Path (ISP)

In a Sudoku puzzle with a single unique solution, any move that does not match that solution immediately creates a board state from which the original puzzle can no longer be solved. Allowing the agent to continue playing on this "poisoned" board introduces flawed data into the training process. The agent might learn spurious correlations from a state that is fundamentally unsolvable.

To prevent this, the environment's `step` function is designed to terminate an episode as soon as the agent makes a move that deviates from the known unique solution.

If we went another way and allowed agent to make wrong moves, we would need to design for correcting or backtracking the wrong moves by providing additional channel in board state that marks clues that are given vs. the agent moves, and change the environment and the reward function to allow changing already made moves. That woul also require allowing much longer games.

#### ISP Implementation

1. **Check Against Solution:** In the `step` function, after an action is taken, it is first compared against the ground-truth solution.
2. **Correct Move:** If the move matches the solution, the agent receives a positive reward, and the episode continues.
3. **Incorrect Move:** If the move does *not* match the solution (even if it's a "valid" placement by Sudoku rules), the agent receives a penalty, and the episode is immediately **terminated**.

This approach ensures that the agent only learns from sequences of moves that are on a valid path to the correct solution, making training more efficient and focused.

### Hindsight Experience Replay (HER)

Standard DQN training involves sampling random past experiences from a large replay buffer. While effective, this method treats all experiences equally and learns from them in a disconnected fashion.

**Hindsight Experience Replay (HER)** is a technique that leverages the outcome of an entire episode to provide more focused training. After an episode concludes, we have a complete trajectory of moves that led to either a success (solved puzzle) or a failure (unsolved board). This complete story is a powerful learning signal.

#### HER Implementation

Our implementation of HER works as follows:

1. **Collect Episode Trajectory:** During an episode, every transition `(state, action, reward, next_state)` is stored in a temporary list, `episode_transitions`.
2. **End-of-Episode Training:** Once the episode finishes (win or lose), we perform an **additional, immediate optimization step**.
3. **Focused Learning:** The `optimize_model` function is called with the entire `episode_transitions` list. This forces the agent to perform a gradient update based on the full sequence of events.
   * For a **successful episode**, this reinforces the entire chain of moves that led to the large final reward.
   * For a **failed episode**, this reinforces the penalties for the incorrect moves that led to a dead end.

This approach provides immediate, contextual feedback to the agent, helping it learn much more quickly which sequences of actions are promising and which are not, rather than waiting for those transitions to be randomly sampled from the buffer over time.

### Residual Connections for Deeper Networks

As the network gets deeper to capture more complex relationships across the Sudoku grid, it can become harder to train due to issues like the vanishing gradient problem. To combat this, we can introduce **Residual Connections** (or skip connections), a core concept from Residual Networks (ResNets).

A residual block allows the network to bypass one or more layers, simply passing the input through. This makes it easier for gradients to flow during backpropagation and allows the model to learn an "identity" function if a block of layers is not useful. For Sudoku, this means we can build a deeper, more powerful CNN that can better integrate local (3x3 box) and global (full grid) information without sacrificing training stability.

### Debugging Insights: Overcoming Training Stagnation

During development, the agent's performance completely stagnated, with the capability score failing to improve over tens of thousands of episodes. And it was happening for all model versions tried - cnn1, cnn2, cnn3, transformer1. A deep dive into the training loop revealed two critical, non-obvious issues:

1. **Silent Computation Graph Corruption:** The most significant bug was an in-place modification of the target Q-values tensor within the `optimize_model` function. Even inside a `with torch.no_grad()` block, altering the tensor directly (`target_q_values[~masks_t] = -1e10`) corrupted the computation graph. This silently prevented gradients from flowing back to the target network, effectively halting all learning. The solution was to replace the in-place operation with an out-of-place one (e.g., `masked_q = target_q_values + additive_mask`), which preserves the graph's integrity and allows learning to proceed.

2. **Data Type Inconsistency:** The state representation was inconsistently handled, switching between `np.int32` and `np.float32` at various stages. This created unnecessary overhead and potential for subtle errors, particularly in functions like `generate_legal_mask` that rely on integer-based logic to build sets. The pipeline was refactored to use `np.int32` for all CPU-side environment logic and `torch.FloatTensor` (via one-hot encoding) exclusively for GPU-side network computations. This created a cleaner, more robust, and more efficient data flow.

### Training Performance Optimization via Vectorization

Initial implementations of the training loop, particularly the `optimize_model` function, suffered from CPU bottlenecks that led to low GPU utilization. This was caused by processing data sequentially within Python loops, forcing the GPU to wait for data.

To resolve this, the data preparation pipeline was completely **vectorized** and re-engineered to operate almost entirely on the GPU. This shift from sequential, per-item processing to parallel, batch processing dramatically reduced CPU overhead and resulted in a **30-50% improvement in training speed**.

The key components of the optimized pipeline are:

1. **Batch-Oriented Helper Functions:** The `state_to_one_hot` and `generate_legal_mask_gpu` functions were refactored to accept and process an entire batch of grids in a single, parallel GPU operation, eliminating slow Python loops.
2. **Elimination of CPU Round-Trips:** The `state_to_one_hot` function was modified to accept GPU tensors directly. This prevents expensive data transfer round-trips (GPU -> CPU -> GPU) that were previously required for data conversion.
3. **Direct Tensor Buffering:** Instead of using list comprehensions and NumPy conversions, transitions are now stacked directly from the replay buffer into a single GPU tensor (`torch.stack`).
4. **GPU-Based Filtering:** The process of filtering out terminal `next_state` values was moved from a CPU-based Python loop to a single, highly optimized boolean indexing operation performed directly on the GPU.

These changes ensure that the entire batch processing pipeline - from sampling the replay buffer to calculating the loss - stays on the GPU, maximizing utilization and allowing the training loop to keep pace with the GPU's computational speed.

### Asynchronous Puzzle Generation

A significant performance bottleneck emerged from the need to generate and validate puzzles on-demand, especially the check for a unique solution, which is a CPU-intensive backtracking task. This caused the highly optimized GPU training pipeline to stall while waiting for the CPU.

To solve this, an **asynchronous puzzle generation** system was implemented using Python's `multiprocessing` module.

1. **Worker Processes:** A pool of background worker processes is spawned, with each worker running on a separate CPU core.
2. **Producer-Consumer Queue:** Workers continuously generate puzzles with random difficulty picked from currently set range, validate them for uniqueness, and place the valid puzzles into a shared queue.
3. **Dynamic Difficulty:** The main training process communicates the desired difficulty (number of clues) to the workers via shared memory (`multiprocessing.Value`). This allows the training curriculum to dynamically request harder or easier puzzles without interrupting the generation flow.

This architecture effectively parallelizes the workload, using idle CPU cores to prepare high-quality training data in advance, ensuring the GPU is never left waiting and maximizing hardware utilization.

## Performance Optimizations

Generating millions of unique Sudoku puzzles for training and validating the agent requires highly optimized utility functions. Significant effort was invested in speeding up two critical components: `count_solutions()` and `get_unique_sudoku()`.

### Optimizing `count_solutions()`

The `count_solutions()` function is essential for verifying that a generated puzzle has exactly one unique solution. A naive backtracking solver can be prohibitively slow, especially on puzzles with many blank cells. The initial implementation, which took over 10 minutes with 64 calls to `count_solutions()` for a difficult generation test case, was optimized to complete the same task in under a second (around 840ms).

This performance gain was achieved through a combination of algorithmic heuristics and data structure improvements:

1. **Minimum Remaining Values (MRV) Heuristic**: Instead of filling blank cells in a fixed order (e.g., top-to-bottom), the solver now intelligently selects the "most constrained" cell - the one with the fewest possible valid numbers. This powerful heuristic dramatically prunes the search tree. By addressing the most difficult cells first, it forces branches of the search that lead to dead ends to fail much earlier, avoiding vast amounts of wasted computation.

2. **O(1) Validity Checks with Sets**: The original solver repeatedly scanned rows, columns, and boxes to check if a move was valid. The optimized version pre-computes and maintains `set` data structures for each row, column, and 3x3 box. This allows for checking the validity of placing a number in a cell in constant O(1) average time (a simple set lookup), a significant improvement over linear scans.

3. **Pre-computation of Empty Cells**: The list of all empty cells is now generated only once at the beginning of the search, rather than being rediscovered on each recursive step. This eliminates redundant scanning of the grid.

### Optimizing `get_unique_sudoku()`

The `get_unique_sudoku()` function generates a puzzle with a specified number of clues by removing cells from a fully solved grid. The core challenge is to remove as many cells as possible while ensuring the puzzle still has only one solution, which requires frequent calls to the expensive `count_solutions()` function. A naive approach of removing cells one by one and checking for uniqueness each time is computationally infeasible.

To solve this, a highly efficient binary search-based algorithm with adaptive chunking was implemented. This approach drastically reduces the number of calls to `count_solutions()`.

1. **Adaptive Chunking (Galloping)**: Instead of removing one cell at a time, the algorithm attempts to remove a large "chunk" of cells at once. The size of this chunk is adaptive: it starts large (e.g., 32 cells) when the grid is mostly full and shrinks as the puzzle becomes more sparse and "fragile." If removing the entire chunk is safe (i.e., the solution remains unique), the algorithm can make huge progress in a single step.

   As the puzzle becomes more difficult (i.e., has fewer clues), the density of "key cells" that are essential for uniqueness increases. To handle this, the chunk size is aggressively reduced to 1 when the number of expensive `count_solutions()` calls grows large (e.g., over 10). This heuristic detects that the puzzle has become "fragile" and switches from a binary search to a more cautious linear scan, which is more efficient than repeatedly failing on large chunks. In special cases with very dense key cells, this saves up to 30% of calls to the solver.

2. **Binary Search on Failure**: When removing a chunk is *not* safe (it introduces multiple solutions), the algorithm doesn't discard the entire chunk. Instead, it performs a binary search *within that chunk* to efficiently find the longest "safe prefix" - the maximum number of cells from the start of the chunk that can be removed together. This allows it to pinpoint the single "key cell" within the chunk that was essential for uniqueness, without having to test every cell individually.

   The binary search includes a further optimization to reduce solver calls. When a prefix is found to be safe, and the remaining part of the chunk being searched is only a single cell, the algorithm can deduce by elimination that this single cell *must* be the key cell. It can then mark it as such and move on without performing an explicit, and redundant, final check on that cell, saving a valuable call to `count_solutions()`.

By combining these two methods with their additional heuristics, the generator can quickly "gallop" through non-essential cells and use a precise binary search to navigate around the critical cells that uphold the puzzle's integrity. This minimizes the number of calls to `count_solutions()` and makes it practical to generate even very difficult puzzles with a low number of clues.

Together with the optimization of `count_solutions()`, the speed of `get_unique_sudoku()` was improved from over 10 minutes to just 840ms for a particularly difficult test case where 64 calls to `count_solution()` were needed to find 24 key cells while generating a puzzle with 54 blanks.
