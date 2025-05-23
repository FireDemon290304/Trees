# Project Overview

This project implements a binary tree and heap data structure in Python. It includes two main classes: `BinaryTree` and `Heap`. The `BinaryTree` class provides methods for managing and visualizing the tree structure, while the `Heap` class extends `BinaryTree` to include heap-specific operations.

## Classes

### BinaryTree
- **Initialization**: Creates a binary tree from a list of values.
- **Methods**:
  - `__getitem__(index)`: Access elements using 1-based indexing.
  - `__setitem__(index, value)`: Set elements in the tree.
  - `length`: Property to get the number of elements in the tree.
  - `height`: Property to calculate the height of the tree.
  - `visualise()`: Visualizes the binary tree structure.

### Heap
- **Initialization**: Creates a heap (max or min) from a list of values.
- **Methods**:
  - `heapify(i)`: Maintains the heap property at index `i`.
  - `build_heap(A)`: Builds a heap from an unordered array.
  - `heapsort(asc=True, arr=None)`: Sorts an array using heap sort.

## Usage

To use the binary tree and heap data structures, you can create instances of the `Heap` class and call the provided methods. For example:

```python
from src.trees_base import Heap

test_input = [4, 1, 3, 2, 16, 9, 10, 14, 8, 7]
heap = Heap(True, test_input)
print(heap)
heap.heapsort(True)
print(heap)
```

## Requirements

Make sure to install the necessary dependencies listed in `requirements.txt` to enable visualization and other functionalities.

## Visualization

The `visualise()` method in the `BinaryTree` class is intended to visualize the tree structure. Currently, it is not implemented, but it can be extended using libraries such as `networkx` and `matplotlib`.

## License

This project is licensed under the MIT License.