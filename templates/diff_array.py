import bisect
from typing import *
from collections import defaultdict


"""
[差分数组(DiffArray)]
支持一维差分、二阶差分、离散差分、二维差分和菱形差分

[时间复杂度]
    1. 一维差分:
        预处理: O(N)  # N为区间数量
        查询: O(1)  # 直接访问差分数组
    2. 离散差分:
        预处理: O(N log N)  # N为需要排序离散点数
        查询: O(log N)  # 二分查找
    3. 二维差分:
        预处理: O(1)
        还原: O(N*M)  # 计算前缀和
    4. 菱形差分:
        预处理: O(1)
        还原: O(S^2)  # S = N+M-1

[空间复杂度]
    1. 一维差分: 
        O(R)  # R为值域范围
    2. 离散差分: 
        O(N)  # 存储离散点
    3. 二维差分: 
        O(N*M)
    4. 菱形差分: 
        O((N+M)^2)

[相关链接]
    1. https://oi-wiki.org/basic/prefix-sum/#差分
"""


"""
codeforces-python: 算法竞赛Python3模板库
#1: 一维差分(数组实现)
https://github.com/xile42/codeforces-python/blob/main/templates/diff_array.py
"""
class DiffArray:

    @staticmethod
    def diff_accumulate(n: int, lrws: List[Tuple[int, int, int]]) -> List[int]:

        diff = [0] * (n + 1)  # 默认下标从0开始(对应l, r)

        for l, r, w in lrws:
            diff[l] += w
            diff[r + 1] -= w

        ans = list()
        cur = 0
        for i in range(n):
            cur += diff[i]
            ans.append(cur)

        return ans  # 默认返回长度n


"""
codeforces-python: 算法竞赛Python3模板库
#2: 一维差分(哈希表实现)
https://github.com/xile42/codeforces-python/blob/main/templates/diff_array.py
"""
class DiffArray:

    @staticmethod
    def valid_points_count(lrws: List[Tuple[int, int, int]], check: Callable[[int], bool] = lambda x: x > 0) -> int:
        """ 计算区间[l, r], 全部更新后, 值满足check的整点数 """

        diff = defaultdict(int)
        for l, r, w in lrws:
            diff[l] += w
            diff[r + 1] -= w

        xs = sorted(diff.keys())
        ans = cur = 0
        for i in range(len(xs)):
            if i > 0 and check(cur):
                ans += xs[i] - xs[i - 1]
            cur += diff[xs[i]]

        return ans

    @staticmethod
    def covered_points_count(lrs: List[Tuple[int, int]]) -> int:
        """ 计算区间[l, r], 全部更新后, 至少被一个区间覆盖的整点数 """

        diff = defaultdict(int)
        for l, r in lrs:
            diff[l] += 1
            diff[r + 1] -= 1

        xs = sorted(diff.keys())
        ans = cur = 0
        for i in range(1, len(xs)):
            if cur > 0:
                ans += xs[i] - xs[i - 1]
            cur += diff[xs[i]]

        return ans

    @staticmethod
    def max_covered_times(lrs: List[Tuple[int, int]]) -> int:
        """ 计算区间[l, r], 全部更新后, 整点最大重叠层数 """

        diff = defaultdict(int)
        for l, r in lrs:
            diff[l] += 1
            diff[r + 1] -= 1

        xs = sorted(diff.keys())
        ans = cur = 0
        for i in range(1, len(xs)):
            cur += diff[xs[i - 1]]
            if cur > ans:
                ans = cur

        return ans

    @staticmethod
    def merge_intervals(lrs: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """ 合并区间 """

        diff = defaultdict(int)
        for l, r in lrs:
            diff[l] += 1
            diff[r + 1] -= 1

        xs = sorted(diff.keys())
        ans = list()
        cur = start = 0
        for i in range(len(xs)):
            if cur > 0 and start == 0:  # 进入覆盖区
                start = xs[i]
            elif cur == 0 and start > 0:  # 离开覆盖区
                ans.append((start, xs[i] - 1))
                start = 0
            cur += diff[xs[i]]

        return ans

    @staticmethod
    def batch_query(lrws: List[Tuple[int, int, int]], queries: List[int]) -> List[int]:
        """ 更新所有操作后, 批量查询对应点的值 """

        diff = defaultdict(int)
        for l, r, w in lrws:
            diff[l] += w
            diff[r + 1] -= w

        xs = sorted(diff.keys())
        prefix = list()
        cur = 0
        for x in xs:
            cur += diff[x]
            prefix.append((x, cur))

        ans = list()
        for q in queries:
            idx = bisect.bisect_right(xs, q) - 1
            ans.append(prefix[idx][1] if idx >= 0 else 0)

        return ans

    @staticmethod
    def diff_accumulate(lrws: List[Tuple[int, int, int]]) -> Tuple[List[int], List[int]]:
        """ 更新所有操作后, 返回所有点的值 """

        diff = defaultdict(int)
        for l, r, w in lrws:
            diff[l] += w
            diff[r + 1] -= w  # 对于两个区间重合端点, 如(l, x)和(x, r), 若二者同时生效, 这里为r + 1, 否则(如人立刻下车)为r(建议传入时r=r-1)

        xs = sorted(diff.keys())
        prefix = list()
        cur = 0
        for x in xs:
            cur += diff[x]
            prefix.append(cur)

        return xs, prefix


# TODO
    # @staticmethod
    # def diff_of_diff(n: int) -> Tuple[
    #     Callable[[int, int, int, int], None],
    #     Callable[[int, int], None],
    #     List[int]
    # ]:
    #     """
    #     二阶差分模板
    #     :param n: 数组长度
    #     :return: (update函数, update2金字塔更新函数, 还原后的数组)
    #     """
    #     diff = [0] * (n + 2)
    #     diff2 = [0] * (n + 2)
    #
    #     def update(l: int, r: int, base: int, step: int) -> None:
    #         """基础更新函数"""
    #         diff[l] += base
    #         # 注意：r+1 可能越界，但初始化时多分配了空间
    #         if r + 1 < len(diff):
    #             diff[r + 1] -= base + step * (r - l)
    #         # 更新二阶差分
    #         if l + 1 < len(diff2):
    #             diff2[l + 1] += step
    #         if r + 1 < len(diff2):
    #             diff2[r + 1] -= step
    #
    #     def update2(i: int, base: int) -> None:
    #         """金字塔式更新：a[j] += max(base - |i-j|, 0)"""
    #         l1 = max(i - base + 1, 0)
    #         update(l1, i, max(base - i, 1), 1)
    #         if base > 1 and i < n - 1:
    #             r1 = min(i + base - 1, n - 1)
    #             update(i + 1, r1, base - 1, -1)
    #
    #     # 还原原数组
    #     sd2, sd = 0, 0
    #     ori = [0] * n
    #     for i in range(n):
    #         sd2 += diff2[i]
    #         sd += diff[i] + sd2
    #         ori[i] = sd
    #
    #     return update, update2, ori
    #
    # @staticmethod
    # def diff_map(a: List[Tuple[int, int]]) -> Tuple[List[Tuple[int, int]], Callable[[int], int]]:
    #     """
    #     离散差分
    #     :param a: 区间列表，每个元素为 (l, r)
    #     :return: (位置-计数值列表, 查询函数)
    #     """
    #     diff = defaultdict(int)
    #     for l, r in a:
    #         diff[l] += 1
    #         diff[r + 1] -= 1
    #
    #     xs = sorted(diff.keys())
    #     pos_and_cnt = []  # (位置, 覆盖计数)
    #     sd = 0
    #     for x in xs:
    #         sd += diff[x]
    #         pos_and_cnt.append((x, sd))
    #
    #     def query(x: int) -> int:
    #         """返回x被多少个区间包含"""
    #         # 二分查找最后一个 <=x 的位置
    #         lo, hi = 0, len(pos_and_cnt) - 1
    #         idx = -1
    #         while lo <= hi:
    #             mid = (lo + hi) // 2
    #             if pos_and_cnt[mid][0] <= x:
    #                 idx = mid
    #                 lo = mid + 1
    #             else:
    #                 hi = mid - 1
    #         return pos_and_cnt[idx][1] if idx != -1 else 0
    #
    #     # 计算被覆盖的整点数量
    #     covered_cnt = 0
    #     for i in range(len(pos_and_cnt) - 1):
    #         if pos_and_cnt[i][1] > 0:
    #             covered_cnt += pos_and_cnt[i + 1][0] - pos_and_cnt[i][0]
    #
    #     return pos_and_cnt, query
    #
    # @staticmethod
    # def diff_2d(n: int, m: int) -> Tuple[
    #     Callable[[int, int, int, int, int], None],
    #     List[List[int]]
    # ]:
    #     """
    #     二维差分
    #     :param n: 行数
    #     :param m: 列数
    #     :return: (更新函数, 结果矩阵)
    #     """
    #     # 初始化 (n+2) x (m+2) 的二维数组
    #     diff = [[0] * (m + 2) for _ in range(n + 2)]
    #
    #     def update(r1: int, c1: int, r2: int, c2: int, val: int) -> None:
    #         """更新子矩阵 [r1, r2] x [c1, c2]"""
    #         diff[r1 + 1][c1 + 1] += val
    #         diff[r1 + 1][c2 + 2] -= val
    #         diff[r2 + 2][c1 + 1] -= val
    #         diff[r2 + 2][c2 + 2] += val
    #
    #     # 还原原矩阵
    #     for i in range(1, n + 1):
    #         for j in range(1, m + 1):
    #             diff[i][j] += diff[i][j - 1] + diff[i - 1][j] - diff[i - 1][j - 1]
    #
    #     # 切出中间 n x m 的结果矩阵
    #     res = [row[1:m + 1] for row in diff[1:n + 1]]
    #     return update, res
    #
    # @staticmethod
    # def diff_rhombus(n: int, m: int, q: int, operations: List[Tuple[int, int, int, int]]) -> List[List[int]]:
    #     """
    #     菱形（曼哈顿距离）差分
    #     :param n: 原矩阵行数
    #     :param m: 原矩阵列数
    #     :param q: 操作数量
    #     :param operations: 操作列表，每个元素为 (x, y, r, val)
    #     :return: 结果矩阵
    #     """
    #     size = n + m - 1  # 变换后矩阵大小
    #     diff = [[0] * (size + 2) for _ in range(size + 2)]
    #
    #     def update(r1: int, c1: int, r2: int, c2: int, val: int) -> None:
    #         """更新变换后的矩形区域"""
    #         diff[r1 + 1][c1 + 1] += val
    #         diff[r1 + 1][c2 + 2] -= val
    #         diff[r2 + 2][c1 + 1] -= val
    #         diff[r2 + 2][c2 + 2] += val
    #
    #     # 处理每个操作
    #     for x, y, r, val in operations:
    #         # 坐标变换 (x, y) -> (x+y, y-x+n-1)
    #         tx, ty = x + y, y - x + n - 1
    #         r1 = max(tx - r, 0)
    #         c1 = max(ty - r, 0)
    #         r2 = min(tx + r, size - 1)
    #         c2 = min(ty + r, size - 1)
    #         update(r1, c1, r2, c2, val)
    #
    #     # 还原变换后的矩阵
    #     for i in range(size):
    #         for j in range(size):
    #             if i > 0:
    #                 diff[i][j] += diff[i - 1][j]
    #             if j > 0:
    #                 diff[i][j] += diff[i][j - 1]
    #             if i > 0 and j > 0:
    #                 diff[i][j] -= diff[i - 1][j - 1]
    #
    #     # 逆变换恢复原矩阵
    #     res = [[0] * m for _ in range(n)]
    #     for i in range(size):
    #         for j in range(size):
    #             # 逆变换条件检查
    #             y = j - (n - 1)  # 还原 y 坐标偏移
    #             if -i <= y <= i and -(2 * n - 2 - i) <= y <= (2 * n - 2 - i) and (i + y) % 2 == 0:
    #                 x = (i - y) // 2
    #                 y = (i + y) // 2
    #                 if 0 <= x < n and 0 <= y < m:
    #                     res[x][y] = diff[i][j]
    #     return res
