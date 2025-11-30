"""
Transformer-based Solver for Sudoku

1. Flattening the Grid: Instead of a 2D image (Batch, Channels, Height, Width),
   a Transformer sees the board as a Sequence of 81 tokens (Batch,
   Sequence_Length, Features).

2. Positional Encodings: A standard convolution knows that pixel (0,0) is next
   to (0,1). A Transformer treats the sequence as an unordered bag of tokens.
   We must add Positional Encodings so the model can learn "Token 0 is
   top-left" and "Token 10 is in the second row".

3. Self-Attention: Instead of fixed geometric kernels (Row/Col convs), the
   model uses Self-Attention to let every cell "look at" every other cell.
   It learns to pay attention to its own row, column, and box neighbors
   automatically.

"""

import torch
import torch.nn as nn

class DQNSolver(nn.Module):
    """
    Transformer-based Solver.
    Treats the Sudoku grid as a sequence of 81 cells.
    Uses Self-Attention to dynamically learn the relationships (Rows, Cols, Boxes)
    between cells rather than using fixed convolutions.
    """
    def __init__(self, _input_shape, _output_size, device, d_model=128, nhead=8, num_layers=4):
        super().__init__()
        self.device = device
        self.d_model = d_model

        # 1. Embedding Layer
        # Input is 10 channels (one-hot digits). We project this to d_model size.
        self.embedding = nn.Linear(10, d_model)

        # 2. Positional Encoding
        # Since we flatten the 9x9 grid to 81 tokens, the model loses spatial awareness.
        # We learn a unique vector for each of the 81 positions.
        self.pos_embedding = nn.Parameter(torch.randn(1, 81, d_model))

        # 3. Transformer Encoder
        # batch_first=True allows input shape (Batch, Seq, Feature)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 4. Final Output Head
        # Projects each of the 81 token embeddings to 9 possible digit scores.
        self.fc = nn.Linear(d_model, 9)

    def forward(self, x):
        """Forward pass."""
        # Input x: (Batch, 10, 9, 9)
        b, c, _h, _w = x.shape

        # 1. Flatten spatial dims: (Batch, 10, 81) -> Permute to (Batch, 81, 10)
        x = x.view(b, c, -1).permute(0, 2, 1)

        # 2. Embed and Add Position
        x = self.embedding(x)  # (Batch, 81, d_model)
        x = x + self.pos_embedding  # Broadcasting adds pos enc to every batch sample

        # 3. Apply Transformer
        x = self.transformer(x)  # (Batch, 81, d_model)

        # 4. Predict
        # Output: (Batch, 81, 9)
        x = self.fc(x)

        # Flatten to standard RL action space: (Batch, 729)
        return x.view(b, -1)
