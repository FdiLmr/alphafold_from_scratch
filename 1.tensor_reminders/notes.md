# Introduction to Tensors in PyTorch


## PyTorch Overview
PyTorch is a **Machine Learning framework** that:
1. Performs tensor operations efficiently (like matrix multiplication) using GPU acceleration.
2. Supports **Automatic Differentiation**, essential for optimizing ML models by computing gradients of error functions.

## Creating Tensors
To create tensors in PyTorch, we use:
- `torch.tensor`: For simple tensors, e.g., vectors or matrices.
- `torch.linspace`: Generates evenly spaced values.
- `torch.ones`: Creates tensors of specified shape filled with a single value.

Basic operations (e.g., `+`, `-`, `<`, `>`, `==`) are supported on tensors, along with data type conversions.

## Tensor Manipulation
### Reshaping
Tensors are stored as 1D arrays in memory. PyTorch’s default **row-major order** means adjacent elements in a row are contiguous in memory. To reshape a tensor, flatten it and adjust the division of elements:
- **Example**: Converting a 2x3 tensor `[[0, 1, 2], [3, 4, 5]]` to 3x2 results in `[[0, 1], [2, 3], [4, 5]]`.

### Indexing and Slicing
PyTorch indexing supports retrieving specific elements and slices:
- **Example**: For a 4x4 transformation matrix `T`, slicing a 3x3 submatrix can be done with `R = T[:3, :3]`.
- The ellipsis `...` operator simplifies handling tensors with varying batch dimensions.

### Computations Along Axes
Functions like `torch.sum`, `torch.argmax`, `torch.mean`, and `torch.std` support computations along specified dimensions (`dim`), with `keepdim=True` allowing retention of singleton dimensions for compatibility.

### Broadcasting
Broadcasting allows operations on tensors of different shapes by expanding singleton dimensions:
- **Example**: Adding vector `B` of shape (3,) to each row of a 4x3 matrix `A`.

### `torch.einsum`
The **Einstein Summation Notation** (`torch.einsum`) simplifies complex operations:
- Matrix multiplication example: `torch.einsum('ik,kj->ij', A, B)` for non-batched, and `torch.einsum('Nik,Nkj->Nij', A, B)` for batched matrix multiplication.


