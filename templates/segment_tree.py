from typing import List, Optional, Callable, DefaultDict
from collections import defaultdict
import math

class SegmentTree:
    """
    A segment tree implementation for range queries and point updates.
    Supports max, min, sum, gcd, etc. operations.
    """

    def __init__(self, data: List[int]) -> None:
        """
        Initialize the segment tree with the given data.
        :param data: List of integers to build the tree from
        """
        self.n = len(data)
        if self.n == 0:
            raise ValueError("Data cannot be empty")
        
        self.size = 1 << (math.ceil(math.log2(self.n)))
        self.tree = [0] * (2 * self.size)
        
        # Initialize leaves
        for i in range(self.n):
            self.tree[self.size + i] = data[i]
        
        # Build the tree
        for i in range(self.size - 1, 0, -1):
            self.tree[i] = self._merge(self.tree[2 * i], self.tree[2 * i + 1])
    
    def _merge(self, a: int, b: int) -> int:
        """
        Merge two segment values. Override this for different operations.
        Default is maximum.
        """
        return max(a, b)
    
    def update(self, index: int, value: int) -> None:
        """
        Update the value at the given index.
        :param index: Index to update (0-based)
        :param value: New value
        """
        if index < 0 or index >= self.n:
            raise IndexError("Index out of bounds")
        
        pos = self.size + index
        self.tree[pos] = value
        
        # Update all ancestors
        pos >>= 1
        while pos >= 1:
            new_val = self._merge(self.tree[2 * pos], self.tree[2 * pos + 1])
            if self.tree[pos] == new_val:
                break
            self.tree[pos] = new_val
            pos >>= 1
    
    def query(self, l: int, r: int) -> int:
        """
        Query the range [l, r] (0-based, inclusive).
        :param l: Left index of query range
        :param r: Right index of query range
        :return: Result of the range query
        """
        if l < 0 or r >= self.n or l > r:
            raise IndexError("Invalid query range")
        
        res = -float('inf')  # Default for max query
        l += self.size
        r += self.size
        
        while l <= r:
            if l % 2 == 1:
                res = self._merge(res, self.tree[l])
                l += 1
            if r % 2 == 0:
                res = self._merge(res, self.tree[r])
                r -= 1
            l >>= 1
            r >>= 1
        
        return res
    
    def query_all(self) -> int:
        """
        Query the entire range.
        :return: Result of the entire range
        """
        return self.tree[1]

class LazySegmentTree:
    """
    A segment tree with lazy propagation for range updates and queries.
    """

    def __init__(self, data: List[int]) -> None:
        """
        Initialize the lazy segment tree with the given data.
        :param data: List of integers to build the tree from
        """
        self.n = len(data)
        if self.n == 0:
            raise ValueError("Data cannot be empty")
        
        self.size = 1 << (math.ceil(math.log2(self.n)))
        self.tree = [0] * (2 * self.size)
        self.lazy = [0] * (2 * self.size)
        
        # Initialize leaves
        for i in range(self.n):
            self.tree[self.size + i] = data[i]
        
        # Build the tree
        for i in range(self.size - 1, 0, -1):
            self.tree[i] = self._merge(self.tree[2 * i], self.tree[2 * i + 1])
    
    def _merge(self, a: int, b: int) -> int:
        """
        Merge two segment values. Default is sum.
        """
        return a + b
    
    def _apply_lazy(self, node: int, l: int, r: int, val: int) -> None:
        """
        Apply lazy update to a node.
        """
        self.tree[node] += val * (r - l + 1)  # For sum queries
        if l != r:
            self.lazy[2 * node] += val
            self.lazy[2 * node + 1] += val
    
    def _push_lazy(self, node: int, l: int, r: int) -> None:
        """
        Push lazy updates to children.
        """
        if self.lazy[node] != 0:
            mid = (l + r) // 2
            self._apply_lazy(2 * node, l, mid, self.lazy[node])
            self._apply_lazy(2 * node + 1, mid + 1, r, self.lazy[node])
            self.lazy[node] = 0
    
    def update_range(self, l: int, r: int, val: int) -> None:
        """
        Update range [l, r] with the given value.
        :param l: Left index of update range (0-based)
        :param r: Right index of update range (0-based)
        :param val: Value to add to the range
        """
        self._update_range(1, 0, self.size - 1, l, r, val)
    
    def _update_range(self, node: int, node_l: int, node_r: int, l: int, r: int, val: int) -> None:
        if r < node_l or l > node_r:
            return
        
        if l <= node_l and node_r <= r:
            self._apply_lazy(node, node_l, node_r, val)
            return
        
        self._push_lazy(node, node_l, node_r)
        mid = (node_l + node_r) // 2
        self._update_range(2 * node, node_l, mid, l, r, val)
        self._update_range(2 * node + 1, mid + 1, node_r, l, r, val)
        self.tree[node] = self._merge(self.tree[2 * node], self.tree[2 * node + 1])
    
    def query_range(self, l: int, r: int) -> int:
        """
        Query range [l, r].
        :param l: Left index of query range (0-based)
        :param r: Right index of query range (0-based)
        :return: Result of the range query
        """
        return self._query_range(1, 0, self.size - 1, l, r)
    
    def _query_range(self, node: int, node_l: int, node_r: int, l: int, r: int) -> int:
        if r < node_l or l > node_r:
            return 0
        
        if l <= node_l and node_r <= r:
            return self.tree[node]
        
        self._push_lazy(node, node_l, node_r)
        mid = (node_l + node_r) // 2
        left = self._query_range(2 * node, node_l, mid, l, r)
        right = self._query_range(2 * node + 1, mid + 1, node_r, l, r)
        return self._merge(left, right)
    
    def query_all(self) -> int:
        """
        Query the entire range.
        :return: Result of the entire range
        """
        return self.tree[1]

class DynamicSegmentTreeNode:
    """
    Node for dynamic segment tree.
    """
    
    def __init__(self, l: int, r: int):
        self.l = l
        self.r = r
        self.left = None
        self.right = None
        self.val = 0
        self.lazy = 0

class DynamicSegmentTree:
    """
    A dynamic segment tree that supports sparse ranges.
    """
    
    def __init__(self, l: int, r: int):
        self.root = DynamicSegmentTreeNode(l, r)
        self.DEFAULT = 0  # Default value for queries
    
    def _push_lazy(self, node: DynamicSegmentTreeNode) -> None:
        if node.lazy != 0:
            if node.l != node.r:
                mid = (node.l + node.r) // 2
                if not node.left:
                    node.left = DynamicSegmentTreeNode(node.l, mid)
                if not node.right:
                    node.right = DynamicSegmentTreeNode(mid + 1, node.r)
                node.left.val += node.lazy * (mid - node.l + 1)
                node.left.lazy += node.lazy
                node.right.val += node.lazy * (node.r - mid)
                node.right.lazy += node.lazy
            node.lazy = 0
    
    def update_range(self, l: int, r: int, val: int) -> None:
        self._update_range(self.root, l, r, val)
    
    def _update_range(self, node: DynamicSegmentTreeNode, l: int, r: int, val: int) -> None:
        if r < node.l or l > node.r:
            return
        
        if l <= node.l and node.r <= r:
            node.val += val * (node.r - node.l + 1)
            node.lazy += val
            return
        
        self._push_lazy(node)
        self._update_range(node.left, l, r, val)
        self._update_range(node.right, l, r, val)
        left_val = node.left.val if node.left else self.DEFAULT
        right_val = node.right.val if node.right else self.DEFAULT
        node.val = left_val + right_val
    
    def query_range(self, l: int, r: int) -> int:
        return self._query_range(self.root, l, r)
    
    def _query_range(self, node: DynamicSegmentTreeNode, l: int, r: int) -> int:
        if not node or r < node.l or l > node.r:
            return self.DEFAULT
        
        if l <= node.l and node.r <= r:
            return node.val
        
        self._push_lazy(node)
        left_val = self._query_range(node.left, l, r) if node.left else self.DEFAULT
        right_val = self._query_range(node.right, l, r) if node.right else self.DEFAULT
        return left_val + right_val
