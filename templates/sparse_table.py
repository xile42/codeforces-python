from math import gcd
from typing import *
from collections import defaultdict


"""
[稀疏表(Sparse Table)]
用于高效查询区间最值(RMQ), 区间GCD等可重复贡献问题
    可重复贡献问题: 对于运算op, 满足 x op x = x, 则称对op运算的区间查询是可重复贡献问题

[算法特性]
1. 下标从0开始，查询区间为左闭右开 [l, r)
2. 支持任意满足结合律且幂等性(重复计算不影响结果)的操作

[时间复杂度]
    预处理: O(NlogN)
    单次查询: O(1)

[空间复杂度]
    O(NlogN)

[相关链接]
    1. https://oi-wiki.org/ds/sparse-table/
"""


"""
codeforces-python: 算法竞赛Python3模板库
#1: 稀疏表(基类)
https://github.com/xile42/codeforces-python/blob/main/templates/sparse_table.py
"""
class SparseTable:

    def __init__(self, arr: List[Union[int, Tuple[int, int]]]) -> None:

        n = len(arr)
        size = n.bit_length()
        self.st = [[0] * size for _ in range(n)]

        for i in range(n):
            self.st[i][0] = arr[i]

        for j in range(1, size):
            for i in range(n - (1 << j) + 1):
                self.st[i][j] = self.op(self.st[i][j - 1], self.st[i + (1 << (j - 1))][j - 1])

    def query(self, l: int, r: int) -> int:
        """ [l, r] """

        k = (r - l + 1).bit_length() - 1
        return self.op(self.st[l][k], self.st[r + 1 - (1 << k)][k])

    def op(self, a: int, b: int) -> int:
        """ 支持的算子包括: min, max, gcd, &, |, product, lca """

        raise NotImplementedError


"""
codeforces-python: 算法竞赛Python3模板库
#2: 稀疏表-MIN运算
https://github.com/xile42/codeforces-python/blob/main/templates/sparse_table.py
"""
class MinSparseTable(SparseTable):

    def op(self, a: int, b: int) -> int:

        return a if a < b else b  # 手写min以提高效率


"""
codeforces-python: 算法竞赛Python3模板库
#3: 稀疏表-MAX运算
https://github.com/xile42/codeforces-python/blob/main/templates/sparse_table.py
"""
class MaxSparseTable(SparseTable):

    def op(self, a: int, b: int) -> int:

        return a if a > b else b  # 手写max以提高效率


"""
codeforces-python: 算法竞赛Python3模板库
#4: 稀疏表-GCD运算
https://github.com/xile42/codeforces-python/blob/main/templates/sparse_table.py
"""
class GcdSparseTable(SparseTable):

    def op(self, a: int, b: int) -> int:

        return gcd(a, b)


"""
codeforces-python: 算法竞赛Python3模板库
#5: 稀疏表-&运算
https://github.com/xile42/codeforces-python/blob/main/templates/sparse_table.py
"""
class BitwiseAndSparseTable(SparseTable):

    def op(self, a: int, b: int) -> int:

        return a & b


"""
codeforces-python: 算法竞赛Python3模板库
#6: 稀疏表-|运算
https://github.com/xile42/codeforces-python/blob/main/templates/sparse_table.py
"""
class BitwiseOrSparseTable(SparseTable):

    def op(self, a: int, b: int) -> int:

        return a | b


"""
codeforces-python: 算法竞赛Python3模板库
#7: 稀疏表-product运算
https://github.com/xile42/codeforces-python/blob/main/templates/sparse_table.py
"""
class ProductSparseTable(SparseTable):

    def __init__(self, arr: List[int], mod: Optional[int] = None):

        self.mod = mod
        super().__init__(arr)

    def op(self, a: int, b: int) -> int:

        return (a * b) % self.mod if self.mod is not None else (a * b)


"""
codeforces-python: 算法竞赛Python3模板库
#8: 稀疏表-带下标版本
https://github.com/xile42/codeforces-python/blob/main/templates/sparse_table.py
"""
class IndexedSparseTable(SparseTable):

    def __init__(self, arr: List[int], op: Optional[Callable] = None):
        self.arr = arr.copy()
        super().__init__(list(zip(arr, range(len(arr)))))

    @staticmethod
    def _default_op(a: Tuple[int, int], b: Tuple[int, int]) -> Tuple[int, int]:

        return a if a[0] < b[0] or (a[0] == b[0] and a[1] < b[1]) else b

    def op(self, a: Tuple[int, int], b: Tuple[int, int]) -> Tuple[int, int]:

        return self._default_op(a, b) if self.op is None else self.op(a, b)
