# -*- coding: utf-8 -*-
"""
Created on Thu May 22 17:35:43 2025

@author: Faith
"""

import math
import networkx as nx
import matplotlib.pyplot as plt

from dataclasses import dataclass, field


class BinaryTree:
    """
    A simple binary array/tree. Indexing starts at 1, following CLRS.
    
    Binary-search-tree property:
    For any node x in the tree, the following holds:
    - If y is a node in the left subtree of then y.key <= x.key
    - If z is a node in the right subtree of then x.key <= z.key
    """
    
    def __init__(self, values=None):
        # self._arr will always be [None, element1, element2, ...]
        # self.use expects an unpadded list of values.
        self._arr = [] # Initialize empty then use self.use
        self.use(values if values is not None else [])  

    def __getitem__(self, index):
        """Get."""
        if index < 1 or index > self.length:
            raise IndexError("Index out of bounds for 1-based indexing.")
        return self._arr[index]
    
#    def __setitem__(self, index, value):
#        """Set."""
#        self._arr[index] = value
        
    def __setitem__(self, index, value):
        """Set. Purely for internal use."""
        if index < 1 or index > self.length: # Or allow extending if index == self.length + 1?
                                             # For now, strict bounds.
            # If padding needed to reach index:
            # while len(self._arr) <= index: self._arr.append(None)
            # Current _arr may be too short if index > self.length
            # For simplicity, assume index is within current [1...length] or use for existing elements
             if index >= len(self._arr): # Need to extend if setting new element beyond current physical padded list
                self._arr.extend([None] * (index - len(self._arr) + 1))
    
        self._arr[index] = value
        
        
    def __repr__(self):
        """Return string representation of the elements (excluding the initial None)."""
        if not self._arr or self.length == 0:
            return "[]"
        return repr(self._arr[1 : self.length + 1]) # Show only actual elements up to self.length


    
    @property
    def length(self):
        """Get number of actual elements in array (consistent with 1-based indexing)."""
        return len(self._arr) - 1 if self._arr else 0 # -1 for the padding None
        
    @property
    def height(self):
        """Get height of the tree (floor(log2(N)))."""
        if self.length == 0:
            return -1 # Convention for empty tree
        return math.floor(math.log2(self.length))

    @property
    def arr(self):
        """Get internal array (including padding). Use with caution."""
        return self._arr


    def use(self, values: list):
        """Replace existing arr with a new one from an unpadded list of values."""
        self._arr = [None] + list(values if values is not None else []) # Ensure values is copied
        return self
    
    def parent(self, i):
        """Find parent index."""
        return i // 2
    
    def left(self, i):
        """Find left child index."""
        return 2 * i
    
    def right(self, i):
        """Find right child index."""
        return 2 * i + 1
    
    
    def visualise(self):
        """Visualise the binary tree using networkx + matplotlib. Root is centered."""
        if self.length == 0:
            print("Tree is empty.")
            return

        G = nx.DiGraph()
        labels = {}

        # Add nodes and edges
        for i in range(1, self.length + 1):
            G.add_node(i)
            labels[i] = str(self.arr[i])
            left = self.left(i)
            right = self.right(i)
            if left <= self.length:
                G.add_edge(i, left)
            if right <= self.length:
                G.add_edge(i, right)

        # Compute positions for a tree layout with root centered
        def hierarchy_pos(G, root=1, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5, pos=None, parent=None):
            if pos is None:
                pos = {root: (xcenter, vert_loc)}
            else:
                pos[root] = (xcenter, vert_loc)
            children = list(G.successors(root))
            if len(children) != 0:
                dx = width / len(children)
                nextx = xcenter - width/2 + dx/2
                for child in children:
                    pos = hierarchy_pos(G, child, width=dx, vert_gap=vert_gap,
                                        vert_loc=vert_loc - vert_gap, xcenter=nextx, pos=pos, parent=root)
                    nextx += dx
            return pos

        pos = hierarchy_pos(G, root=1, width=1.0, xcenter=0.5)

        plt.figure(figsize=(8, 5))
        nx.draw(G, pos, with_labels=False, arrows=False, node_size=1200, node_color="#90caf9")
        nx.draw_networkx_labels(G, pos, labels, font_size=12, font_color="black")
        plt.title("Binary Tree Visualisation")
        plt.axis('off')
        plt.show()



class Heap(BinaryTree):
    """
    A collection of min and max heap operations.

    Can sattisfy two properties:
        - Max heap property: For all node i > 1 (i.e., excluding the root) A[parent(i)] >= A[i]
        - Min heap property: For all node i > 1 (i.e., excluding the root) A[parent(i)] <= A[i]

    A heap has two attributes:
        - length = the number of elements in the array
        - heap-size = number of elements of the heap stored in the array
        - 0 <= heapsize <= length
        
    A heap has two attributes from CLRS perspective:
        - A.length = the number of elements in the array storing the heap
        - A.heap-size = number of elements of the heap stored in the array
        - 0 <= A.heap-size <= A.length
    Here, self.length is A.length, and self.heap_size is A.heap-size.

    """
    
    def __init__(self, is_max_heap: bool = True, values: list = None):
        super().__init__(values)
        self.heap_size: int = self.length
        self.is_max = is_max_heap
        if values is not None: # If values were provided, build heap on them
             self.build_heap(values)
    
    # TODO: For deep trees make iterative
    def heapify(self, i) -> None:
        """Maintain the max heap property at index i. Runtime: O(lg n)."""
        l = self.left(i)
        r = self.right(i)
        
        def compare(x, y):
            return x > y if self.is_max else x < y
        
        best = i
        
        # Find best
        if l <= self.heap_size and compare(self.arr[l], self.arr[i]):
            best = l
        if r <= self.heap_size and compare(self.arr[r], self.arr[best]):
            best = r
        
        if best != i:
            # Swap
            self.arr[i], self.arr[best] = self.arr[best], self.arr[i]
            self.heapify(best)   # Recursion

    def build_heap(self, A: list) -> None:
        """Produce a max-heap from an unordered input array. Runtime: O(n)."""
        if not isinstance(A, BinaryTree):
            self.use(A)
        else:
            # Set new internal list
            self._arr = A.copy()
            
        self.heap_size = self.length
        
        for i in range(self.length // 2, 0, -1):
            self.heapify(i)
    
    # TODO: Currently modifies list in place. Add toggle for copy, so heap property remains after call
    def heapsort(self, asc=True, arr=None) -> None:
        """
        Sorts an array.
        
        If arr_values is provided, it sorts that list (and updates the heap's internal array).
        Otherwise, it sorts the heap's current internal array.
        'asc = True' for ascending sort (builds and uses a max-heap).
        'asc = False' for descending sort (builds and uses a min-heap).
        """
        self.is_max = True if asc else False
        
        self.build_heap(self.arr if arr is None else arr)
        
        for i in range(self.length, 1, -1): # Order matters!
            self.arr[1], self.arr[i] = self.arr[i], self.arr[1] # Decreasing heap-size = focus on the tree representation of the subarray A[1..heap-size]
            self.heap_size -= 1
            self.heapify(1)

# TODO: BST and RBT. Prerequisite is making node class. Use @dataclass

@dataclass
class TreeNode:
    key: int
    left: "TreeNode" = None
    right: "TreeNode" = None
    parent: "TreeNode" = None
    colour: str = field(default='red')  # RBT

    def __repr__(self) -> str:
        return f"TreeNode({self.key})"


class BST:
    def __init__(self):
        self.root = None

    def insert(self, key):
        pass

    def inorder_walk(self):
        pass

    def search(self, key, node=None):
        pass


class RBT:
    pass

if __name__ == '__main__':
    test_input = [4, 1, 3, 2, 16, 9, 10, 14, 8, 7]
    
    print(test_input)
    
    heap = Heap(True, test_input)
    print(heap)
    
    #heap.heapsort(True, test_input)
    #print(heap)
    #heap.heapsort(False, test_input)
    #print(heap)
    
    heap.visualise()


