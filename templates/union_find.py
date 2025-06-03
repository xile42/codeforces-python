from typing import *
from collections import defaultdict


"""
[并查集(UnionFind)]
相关内容: 路径压缩 按秩合并(启发式合并)
相关链接: 
    1. https://oi-wiki.org/ds/dsu/
"""

"""
CodeForces题单: (by 灵茶山艾府 https://github.com/EndlessCheng/codeforces-go/tree/master )

    [一般]
    1. https://codeforces.com/problemset/problem/755/C 1300
    2. https://codeforces.com/problemset/problem/1167/C 1400
    3. https://codeforces.com/problemset/problem/2060/E 1500
    4. https://codeforces.com/problemset/problem/1209/D 1700
    5. https://codeforces.com/problemset/problem/1411/C 1700
    6. https://codeforces.com/problemset/problem/371/D 1800
    7. https://codeforces.com/problemset/problem/87/D 2300
    8. https://codeforces.com/problemset/problem/1726/D 2000 处理图上的环
    9. https://codeforces.com/problemset/problem/1851/G 2000 离线

    [质因子并查集]
    - 预处理质因子（见 math.go 中的 primeDivisorsAll）
    - 枚举 a[i] 的质因子 p, 用 pre[p] 记录质因子上一次出现的下标(初始化成 -1), 然后 merge(i, pre[p]) if pre[p] > 0
    1. https://codeforces.com/contest/1978/problem/F 2400

    [数组标记/区间合并相关]
    - 经典模型是一维区间覆盖染色, 通过倒序+并查集解决
    1. https://codeforces.com/problemset/problem/1791/F 1500
    2. https://codeforces.com/problemset/problem/1041/C 1600
    3. https://codeforces.com/problemset/problem/827/A 1700
    4. https://codeforces.com/problemset/problem/1157/E 1700
    5. https://codeforces.com/problemset/problem/724/D 1900
    6. https://codeforces.com/problemset/problem/2018/D 2200

    [树+点权/边权的顺序]
    1. https://codeforces.com/problemset/problem/87/D 2300
    2. https://codeforces.com/problemset/problem/915/F 2400 贡献法

    [其他]
    1. https://codeforces.com/problemset/problem/371/D 1800 接水问题
    2. https://codeforces.com/problemset/problem/292/D 1900
    3. https://codeforces.com/problemset/problem/566/D 1900 任意合并+区间合并
    4. https://codeforces.com/contest/1494/problem/D 2300 动态加点
    5. https://codeforces.com/problemset/problem/1012/B 1900
    6. https://codeforces.com/problemset/problem/1466/F 2100
    7. https://codeforces.com/problemset/problem/455/C 2100
    8. https://codeforces.com/problemset/problem/292/D 1900 前缀和 后缀和
    9. https://codeforces.com/problemset/problem/859/E 2100 维护树或基环树
    10. https://codeforces.com/problemset/problem/650/C 2200 求矩阵的 rank 矩阵
    11. https://codeforces.com/problemset/problem/1253/D 1700 转换
    12. https://codeforces.com/contest/1851/problem/G 2000 能力守恒+离线
    13. https://codeforces.com/contest/884/problem/E 2500
    14. https://codeforces.com/problemset/problem/1416/D 2600 DSU 重构树
"""


"""
codeforces-python: 算法竞赛Python3模板库
#1: 基础并查集
https://github.com/xile42/codeforces-python/blob/main/templates/union_find.py
"""
class UnionFind:

    def __init__(self, n: int) -> None:

        self.root = list(range(n))
        self.size = [1] * n  # 并查集大小/秩
        self.part = n  # 连通分量数

    # # 递归写法
    # def find(self, x):
    #
    #     if x != self.root[x]:
    #         self.root[x] = self.find(self.root[x])
    #
    #     return self.root[x]

    # 非递归写法, 效率更优
    def find(self, x: int) -> int:

        root = x
        while self.root[root] != root:
            root = self.root[root]
        while self.root[x] != root:
            self.root[x], x = root, self.root[x]

        return root

    def merge(self, x: int, y: int) -> Optional[int]:

        root_x, root_y = self.find(x), self.find(y)
        if root_x == root_y:
            return None  # 未发生合并
        # 按秩合并
        if self.size[root_x] >= self.size[root_y]:
            root_x, root_y = root_y, root_x
        self.root[root_x] = root_y
        self.size[root_y] += self.size[root_x]
        self.size[root_x] = 0
        self.part -= 1

        return root_y

    def is_merged(self, x: int, y: int) -> bool:

        return self.find(x) == self.find(y)

    def get_parts(self):

        parts = defaultdict(list)
        n = len(self.root)
        for i in range(n):
            parts[self.find(i)].append(i)

        return parts

    def get_sizes(self):

        sizes = defaultdict(int)
        n = len(self.root)
        for i in range(n):
            sizes[self.find(i)] = self.size[self.find(i)]

        return sizes
