import math
import bisect
from typing import *
from collections import defaultdict


"""
[前缀和(PrefixSum)]
支持一维前缀和, 同余前缀和, 循环前缀和, 带权前缀和, 拆位前缀和, 二维前缀和, 斜向前缀和, 行/列前缀和, 三角形/金字塔/菱形前缀, 贡献和计算

[算法特性]
    1. 前缀和数组通常从1开始索引, 避免边界检查
    2. 区间查询默认左闭右闭原则 [l,r], 即prefix_sum[r + 1] - prefix_sum[l]
    3. 二维前缀和使用容斥原理计算矩形区域和
    4. 斜向前缀和适用于菱形区域查询
    5. 贡献和计算利用排序后每个元素的相对位置关系

[时间复杂度]
    1. 基础前缀和:
        预处理: O(N)  # N为数组长度
        查询: O(1)  # 直接计算区间和
    2. 同余前缀和:
        预处理: O(N+M)  # M为模数
        查询: O(1)
    3. 循环前缀和:
        预处理: O(N)
        查询: O(1)
    4. 带权前缀和:
        预处理: O(N)
        查询: O(1)
    5. 拆位前缀和:
        预处理: O(N*K)  # K为最大位数
        查询AND/OR: O(K)
    6. 二维前缀和:
        预处理: O(N*M)
        查询: O(1)
    7. 斜向前缀和:
        预处理: O(N*M)
        查询: O(1)
    8. 行/列前缀和:
        预处理: O(N*M)
        查询: O(1)
    9. 三角形/菱形前缀和:
        预处理: O(N*M)
        查询: O(1)
    10. 贡献和计算:
        预处理: O(N log N)  # 需要排序
        计算: O(N)

[空间复杂度]
    1. 基础前缀和: 
        O(N)
    2. 同余前缀和: 
        O(N+M)
    3. 循环前缀和: 
        O(N)
    4. 带权前缀和: 
        O(N)
    5. 拆位前缀和: 
        O(N*K)
    6. 二维前缀和: 
        O(N*M)
    7. 斜向前缀和: 
        O(N*M)
    8. 行/列前缀和: 
        O(N*M)
    9. 三角形/菱形前缀和: 
        O(N*M)
    10. 贡献和计算: 
        O(N)

[相关链接]
    1. https://oi-wiki.org/basic/prefix-sum/#前缀和
"""


"""
codeforces-python: 算法竞赛Python3模板库
#1: 一维前缀和
https://github.com/xile42/codeforces-python/blob/main/templates/prefix_sum.py
"""
class PrefixSum:

    def __init__(self, arr: List[int]):

        self.n = len(arr)
        self.prefix = [0] * (self.n + 1)

        for i in range(self.n):
            self.prefix[i + 1] = self.prefix[i] + arr[i]

    def query(self, l: int, r: int) -> int:
        """ [l, r] """
        return self.prefix[r + 1] - self.prefix[l]  # 忽略越界检查


"""
codeforces-python: 算法竞赛Python3模板库
#2: 一维距离和
https://github.com/xile42/codeforces-python/blob/main/templates/prefix_sum.py
"""
class DistancePrefixSum:

    def __init__(self, arr: List[int], is_sorted: bool = False):

        self.n = len(arr)
        self.arr = sorted(arr) if not is_sorted else arr
        self.prefix = [0] * (self.n + 1)

        for i in range(self.n):
            self.prefix[i + 1] = self.prefix[i] + self.arr[i]

    def query(self, target: Optional[int]=None, l: int=0, r: Optional[int]=None) -> int:
        """ sum(abs(a[i] - target)) for i in [l, r] """

        target = self.arr[self.n // 2] if target is None else target
        r = self.n - 1 if r is None else r
        i = bisect.bisect_left(self.arr, target, l, r + 1)
        left = target * (i - l) - (self.prefix[i] - self.prefix[l])
        right = (self.prefix[r + 1] - self.prefix[i]) - target * (r + 1 - i)

        return left + right


"""
codeforces-python: 算法竞赛Python3模板库
#3: 一维同余前缀和
https://github.com/xile42/codeforces-python/blob/main/templates/prefix_sum.py
"""
class ModPrefixSum:

    def __init__(self, arr: List[int], mod: int):

        self.mod = mod
        self.n = len(arr)
        self.prefix = [0] * (self.n + self.mod)

        for i in range(self.n):
            self.prefix[i + self.mod] = self.prefix[i] + arr[i]

    def query(self, l: int, r: int, rem: int) -> int:
        """ [l, r], i ≡ rem (mod mod)"""

        def pre(x: int) -> int:

            if x % self.mod <= rem:
                return self.prefix[x // self.mod * self.mod + rem]
            return self.prefix[((x + self.mod - 1) // self.mod) * self.mod + rem]

        return pre(r + 1) - pre(l)


"""
codeforces-python: 算法竞赛Python3模板库
#4: 一维无限循环前缀和
https://github.com/xile42/codeforces-python/blob/main/templates/prefix_sum.py
"""
class LoopPrefixSum:

    def __init__(self, arr: List[int]):

        self.n = len(arr)
        self.prefix = [0] * (self.n + 1)

        for i in range(self.n):
            self.prefix[i + 1] = self.prefix[i] + arr[i]
        self.total = self.prefix[self.n]  # 数组总和

    def query(self, l: int, r: int) -> int:
        """ [l, r] """

        def pre(x: int) -> int:

            return self.total * (x // self.n) + self.prefix[x % self.n]  # 注意可能转为大整数运算, 必要时取模

        return pre(r + 1) - pre(l)


# TODO
# class PrefixSum:
#
#     @staticmethod
#     def weighted_prefix_sum(arr: List[int]) -> List[List[int]]:
#         """带权前缀和，权重是等差数列"""
#         n = len(arr)
#         sum_ = [0] * (n + 1)
#         i_sum = [0] * (n + 1)
#         for i in range(n):
#             sum_[i + 1] = sum_[i] + arr[i]
#             i_sum[i + 1] = i_sum[i] + (i + 1) * arr[i]
#         return sum_, i_sum
#
#     @staticmethod
#     def bit_sum(arr: List[int]) -> List[List[int]]:
#         """拆位前缀和"""
#         max_bit = max(arr).bit_length() if arr else 0
#         n = len(arr)
#         sum_ = [[0] * max_bit for _ in range(n + 1)]
#         for i in range(n):
#             for j in range(max_bit):
#                 sum_[i + 1][j] = sum_[i][j] + ((arr[i] >> j) & 1)
#         return sum_
#
#
# class MatrixPrefixSum:
#
#     @staticmethod
#     def matrix_sum_2d(matrix: List[List[int]]) -> List[List[int]]:
#         """二维前缀和"""
#         n, m = len(matrix), len(matrix[0]) if matrix else 0
#         prefix = [[0] * (m + 1) for _ in range(n + 1)]
#         for i in range(n):
#             for j in range(m):
#                 prefix[i + 1][j + 1] = prefix[i + 1][j] + prefix[i][j + 1] - prefix[i][j] + matrix[i][j]
#         return prefix
#
#     @staticmethod
#     def diagonal_sum(matrix: List[List[int]]) -> List[List[int]]:
#         """矩阵斜向前缀和"""
#         n, m = len(matrix), len(matrix[0]) if matrix else 0
#         ds = [[0] * (m + 1) for _ in range(n + 1)]  # 主对角线方向 ↘
#         as_ = [[0] * (m + 1) for _ in range(n + 1)]  # 反对角线方向 ↙
#         for i in range(n):
#             for j in range(m):
#                 ds[i + 1][j + 1] = ds[i][j] + matrix[i][j]
#                 as_[i + 1][j] = as_[i][j + 1] + matrix[i][j]
#         return ds, as_
#
#     @staticmethod
#     def row_col_sum(matrix: List[List[int]]) -> List[List[List[int]]]:
#         """矩阵每行每列的前缀和"""
#         n, m = len(matrix), len(matrix[0]) if matrix else 0
#         row_sum = [[0] * (m + 1) for _ in range(n)]
#         col_sum = [[0] * m for _ in range(n + 1)]
#
#         for i in range(n):
#             for j in range(m):
#                 row_sum[i][j + 1] = row_sum[i][j] + matrix[i][j]
#
#         for j in range(m):
#             for i in range(n):
#                 col_sum[i + 1][j] = col_sum[i][j] + matrix[i][j]
#
#         return row_sum, col_sum
#
#
# class TrianglePrefixSum:
#
#     @staticmethod
#     def right_triangle_sum(matrix: List[List[int]]) -> List[List[List[int]]]:
#         """等腰直角三角形区域和"""
#         n, m = len(matrix), len(matrix[0]) if matrix else 0
#
#         # 矩形前缀和
#         rect_sum = [[0] * (m + 1) for _ in range(n + 1)]
#         for i in range(n):
#             for j in range(m):
#                 rect_sum[i + 1][j + 1] = rect_sum[i + 1][j] + rect_sum[i][j + 1] - rect_sum[i][j] + matrix[i][j]
#
#         # 四类三角形前缀和
#         ts1 = [[0] * (m + 1) for _ in range(n + 1)]  # ◣
#         ts2 = [[0] * (m + 1) for _ in range(n + 1)]  # ◢
#         ts3 = [[0] * (m + 1) for _ in range(n + 1)]  # ◥
#         ts4 = [[0] * (m + 1) for _ in range(n + 1)]  # ◤
#
#         for i in range(n):
#             s = 0
#             for j in range(m):
#                 s += matrix[i][j]
#                 ts1[i + 1][j + 1] = ts1[i][j] + s
#                 if j >= i:
#                     s -= matrix[i][j - i]
#
#         for i in range(n):
#             s = 0
#             for j in range(m - 1, -1, -1):
#                 s += matrix[i][j]
#                 ts2[i + 1][j] = ts2[i][j + 1] + s
#                 if (m - 1 - j) >= i:
#                     s -= matrix[i][j + i]
#
#         for i in range(n - 1, -1, -1):
#             s = 0
#             for j in range(m - 1, -1, -1):
#                 s += matrix[i][j]
#                 ts3[i][j] = ts3[i + 1][j + 1] + s
#                 if (m - 1 - j) >= (n - 1 - i):
#                     s -= matrix[i][j + (n - 1 - i)]
#
#         for i in range(n - 1, -1, -1):
#             s = 0
#             for j in range(m):
#                 s += matrix[i][j]
#                 ts4[i][j + 1] = ts4[i + 1][j] + s
#                 if j >= (n - 1 - i):
#                     s -= matrix[i][j - (n - 1 - i)]
#
#         return rect_sum, ts1, ts2, ts3, ts4
#
#
# class ContributionSum:
#
#     @staticmethod
#     def calculate(arr: List[int]) -> int:
#         """利用每个数产生的贡献计算∑|ai-aj|, i≠j"""
#         arr.sort()
#         total = 0
#         n = len(arr)
#         for i in range(n):
#             total += arr[i] * (2 * i + 1 - n)
#         return total
