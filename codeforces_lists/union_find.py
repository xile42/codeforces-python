"""
CodeForces Union Find 题单:

    [一般]
    1. https://codeforces.com/problemset/problem/755/C 1300  [√ Accecpt using template #1]
    2. https://codeforces.com/problemset/problem/1167/C 1400
    3. https://codeforces.com/problemset/problem/2060/E 1500
    4. https://codeforces.com/problemset/problem/1209/D 1700
    5. https://codeforces.com/problemset/problem/1411/C 1700
    6. https://codeforces.com/problemset/problem/371/D 1800
    7. https://codeforces.com/problemset/problem/87/D 2300
    8. https://codeforces.com/problemset/problem/1726/D 2000 处理图上的环
    9. https://codeforces.com/problemset/problem/1851/G 2000 离线

    [轮换]
    - 任意一个置换, 可以分解为若干个不相交的轮换
    - 结论: 将任意排列通过置换恢复到原始排列所需的最少对换次数为 n − k, 其中 n 是排列的长度, k 是置换分解中的轮换数量(包括长度为 1 的轮换)
    1. https://codeforces.com/problemset/problem/1249/B1 1000  [√ Accecpt using template #1]
    2. https://codeforces.com/problemset/problem/1249/B2 1300  [√ Accecpt using template #1]
    3. https://codeforces.com/problemset/problem/2008/D 1100  [√ Accecpt using template #1]

    [坐标系]
    - 一般是x, y坐标上进行并查集合并操作
    - 往往需要做辅助 offset, 如对于一个点 i (x, y), 可以合并点 i 与 N + x, N + offset + y, 其中 N 为点数, offset 为一个常数偏移量
    1. https://codeforces.com/problemset/problem/217/A 1200  [√ Accecpt using template #2]

    [数组并查集]
    - 将数组下标作为并查集的节点, 通过合并相邻的元素(一般左合并到右)来实现一些操作
    1. https://codeforces.com/problemset/problem/1927/D 1300  [√ Accecpt using template #3]

    [质因子并查集]
    - 预处理质因子(见 math.py 中的 #1 埃氏筛(EratosthenesSieve) 与 #2 欧拉筛(EulerSieve))
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
