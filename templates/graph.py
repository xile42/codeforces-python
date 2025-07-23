from math import *
from heapq import *
from typing import *
from collections import *


"""
[拓扑排序(Topological Sort)]
基于Kahn算法实现, 通过不断移除入度为0的节点来确定拓扑顺序

[时间复杂度]
    O(V + E)  # 其中V为顶点数，E为边数
[空间复杂度]
    O(V + E)  # 邻接表存储图

[相关链接]
    1. https://oi-wiki.org/graph/topo/
    2. https://en.wikipedia.org/wiki/Topological_sorting#Kahn's_algorithm
"""


"""
codeforces-python: 算法竞赛Python3模板库
#1: 拓扑排序
https://github.com/xile42/codeforces-python/blob/main/templates/graph.py
"""
class TopologicalSort:

    def __init__(self, n: int, edges: List[List[int]]) -> None:

        self.n = n
        self.g = [[] for _ in range(n)]
        self.in_deg = [0] * n
        for x, y in edges:
            self.g[x].append(y)
            self.in_deg[y] += 1

    def sort(self) -> Optional[List[int]]:
        """ 返回拓扑排序结果，若存在环则返回None """

        topo_order = list()
        q = deque(i for i, d in enumerate(self.in_deg) if d == 0)

        while q:
            x = q.popleft()
            topo_order.append(x)
            for y in self.g[x]:
                self.in_deg[y] -= 1
                if self.in_deg[y] == 0:
                    q.append(y)

        if len(topo_order) < self.n:  # 存在环
            return None

        return topo_order

    def has_cycle(self) -> bool:
        """ 判断图中是否存在环 """

        return self.sort() is None

    def get_longest_path(self) -> int:
        """ 返回DAG中的最长路径长度 """

        topo_order = self.sort()
        if topo_order is None:
            return -1  # 存在环

        dist = [0] * self.n
        for x in topo_order:
            for y in self.g[x]:
                if dist[y] < dist[x] + 1:
                    dist[y] = dist[x] + 1

        return max(dist)


"""
[单源最短路径(Dijkstra)]
适用于无负权边的图, 使用优先队列(最小堆)优化

[时间复杂度]
    O(M log M)  # 其中M为边数，使用优先队列实现
[空间复杂度]
    O(N + M)    # 其中N为节点数，M为边数

[特性]
    1. 仅适用于无负权边的图
    2. 单源最短路径
    3. 使用贪心算法思想
    4. 优先队列优化实现

[相关链接]
    1. https://oi-wiki.org/graph/shortest-path/#dijkstra
    2. https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm
"""


"""
codeforces-python: 算法竞赛Python3模板库
#2: Dijkstra
https://github.com/xile42/codeforces-python/blob/main/templates/graph.py
"""
class Dijkstra:

    def __init__(self, n: int, edges: List[List[int]], start: int, directed: bool = True) -> None:

        self.n = n
        self.g = [[] for _ in range(n)]
        for u, v, w in edges:
            self.g[u].append((v, w))
            if not directed:
                self.g[v].append((u, w))

        self.start = start
        self.directed = directed
        self.dis = [inf] * self.n
        self._run()

    def _run(self) -> None:
        """ 执行Dijkstra算法计算最短路径 """

        self.dis[self.start] = 0
        heap = [(0, self.start)]

        while heap:
            dis_x, x = heappop(heap)
            if dis_x > self.dis[x]:
                continue
            for y, w in self.g[x]:
                new_dis_y = dis_x + w
                if new_dis_y < self.dis[y]:
                    self.dis[y] = new_dis_y
                    heappush(heap, (new_dis_y, y))

    def get_dis(self) -> List[float]:
        """ 获取起点到所有节点的最短距离列表 """

        return self.dis

    def get_dis_to_tar(self, target: int) -> float:
        """ 获取起点到目标节点的最短距离 """

        return self.dis[target]

    def get_path(self, target: int) -> Optional[List[int]]:
        """获取起点到目标节点的最短路径(需要额外预处理)"""
        # 注意：原始Dijkstra实现不记录路径，需要添加前驱节点记录
        raise NotImplementedError("需要添加前驱节点记录以实现此功能")

    def get_reachable_nodes(self) -> List[int]:
        """ 获取所有可达节点的列表 """

        return [i for i, d in enumerate(self.dis) if d != inf]

    def get_unreachable_nodes(self) -> List[int]:
        """ 获取所有不可达节点的列表 """

        return [i for i, d in enumerate(self.dis) if d == inf]

    def is_reachable(self, target: int) -> bool:
        """ 判断目标节点是否可达 """

        return self.dis[target] != inf
