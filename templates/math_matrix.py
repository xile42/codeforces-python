from typing import *


"""
codeforces-python: 算法竞赛Python3模板库
#1: 矩阵基础操作
https://github.com/xile42/codeforces-python/blob/main/templates/math_matrix.py
"""
class MatrixBasics:

    @staticmethod
    def read_matrix(n: int) -> List[List[int]]:

        matrix = []
        for _ in range(n):
            row = list(map(int, input().split()))
            matrix.append(row)

        return matrix

    @staticmethod
    def copy_matrix(matrix: List[List[int]]) -> List[List[int]]:

        return [row[:] for row in matrix]

    @staticmethod
    def rotate_matrix(matrix: List[List[int]]) -> List[List[int]]:

        n, m = len(matrix), len(matrix[0])
        ans = [[0] * n for _ in range(m)]
        for j in range(m):
            for i in range(n):
                ans[j][n - 1 - i] = matrix[i][j]

        return ans


"""
[矩阵快速幂(Matrix Exponentiation)]
用于高效计算矩阵的幂运算, 常用于线性递推关系的快速计算(如斐波那契数列)和DP优化(尤其状态机DP)
支持模运算以防止数值溢出

[时间复杂度]
    add/sub: O(N^2)  # 矩阵加减法
    mul: O(N^3)  # 朴素矩阵乘法
    pow_mul: O(logK * N^3)  # 快速幂, K为幂次
[空间复杂度]
    O(N^2)  # 存储矩阵

[相关链接]
    1. https://oi-wiki.org/math/linear-algebra/matrix/
"""


"""
codeforces-python: 算法竞赛Python3模板库
#2: 矩阵快速幂&运算
https://github.com/xile42/codeforces-python/blob/main/templates/math_matrix.py
"""
class Matrix:

    def __init__(self, matrix: List[List[int]], mod: Optional[int] = None) -> None:

        self.matrix = matrix
        self.mod = mod
        self.rows = len(matrix)
        self.cols = len(matrix[0]) if self.rows > 0 else 0  # 忽略列数相等检查

    @classmethod
    def new_matrix(cls, n: int, m: int, mod: Optional[int] = None) -> "Matrix":

        return Matrix([[0] * m for _ in range(n)], mod)

    def swap_rows(self, i: int, j: int) -> None:

        if not (0 <= i < self.rows and 0 <= j < self.rows):
            raise ValueError("Row index out of range")

        self.matrix[i], self.matrix[j] = self.matrix[j], self.matrix[i]

    def swap_cols(self, i: int, j: int) -> None:

        if not (0 <= i < self.cols and 0 <= j < self.cols):
            raise ValueError("Column index out of range")

        for row in self.matrix:
            row[i], row[j] = row[j], row[i]

    def mul_row(self, i: int, k: int) -> None:

        if not 0 <= i < self.rows:
            raise ValueError("Row index out of range")

        for j in range(self.cols):
            self.matrix[i][j] *= k
            if self.mod is not None:
                self.matrix[i][j] %= self.mod

    def trace(self) -> int:

        if self.rows != self.cols:
            raise ValueError("Trace is only defined for square matrices")

        trace = 0
        for i in range(self.rows):
            trace += self.matrix[i][i]
            if self.mod is not None:
                trace %= self.mod

        return trace

    def __matmul__(self, other: "Matrix") -> "Matrix":
        """ self @ other """

        if self.cols != other.rows:
            raise ValueError(f"Matrix dimension mismatch: {self.rows}x{self.cols} * {other.rows}x{other.cols}")

        ans = Matrix.new_matrix(self.rows, other.cols, self.mod)

        for i in range(self.rows):
            for k in range(self.cols):
                if self.matrix[i][k] == 0:
                    continue  # 稀疏矩阵优化
                for j in range(other.cols):
                    ans.matrix[i][j] += self.matrix[i][k] * other.matrix[k][j]
                    if self.mod is not None:
                        ans.matrix[i][j] %= self.mod

        return ans

    def __add__(self, other: 'Matrix') -> 'Matrix':
        """ self + other """

        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError(f"Matrix dimension mismatch: {self.rows}x{self.cols} + {other.rows}x{other.cols}")

        ans = Matrix.new_matrix(self.rows, self.cols, self.mod)

        for i in range(self.rows):
            for j in range(self.cols):
                ans.matrix[i][j] = self.matrix[i][j] + other.matrix[i][j]
                if self.mod is not None:
                    ans.matrix[i][j] %= self.mod

        return ans

    def __sub__(self, other: 'Matrix') -> 'Matrix':
        """ self - other """

        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError(f"Matrix dimension mismatch: {self.rows}x{self.cols} - {other.rows}x{other.cols}")

        ans = Matrix.new_matrix(self.rows, self.cols, self.mod)

        for i in range(self.rows):
            for j in range(self.cols):
                ans.matrix[i][j] = self.matrix[i][j] - other.matrix[i][j]
                if self.mod is not None:
                    ans.matrix[i][j] = (ans.matrix[i][j] % self.mod + self.mod) % self.mod  # 确保结果为非负数

        return ans

    def pow_mul(self, n: int, f0: Optional["Matrix"] = None) -> "Matrix":
        """ (matrix ^ n) * f0; 若f0 is None: matrix ^ n """

        if f0 is None:

            if self.rows != self.cols:
                raise ValueError("Only square matrices can be raised to a power")
            f0 = Matrix.new_matrix(self.rows, self.rows, self.mod)
            for i in range(self.rows):
                f0.matrix[i][i] = 1  # 初始化为单位矩阵

        elif self.cols != f0.rows:

            raise ValueError(f"Matrix dimension mismatch: {self.rows}x{self.cols} ^ {n} * {f0.rows}x{f0.cols}")

        cur = self
        ans = f0

        while n > 0:
            if n & 1:
                ans = cur @ ans
            cur = cur @ cur
            n >>= 1

        return ans

    def __str__(self) -> str:

        return "\n".join([" ".join(map(str, row)) for row in self.matrix])

    def __eq__(self, other: object) -> bool:

        if not isinstance(other, Matrix):
            return False

        return self.matrix == other.matrix



# ↓TODO: 风格修改


mod = 10**9 + 7  # 根据需要设置模数


# 一般是状态机 DP
# 操作 k 次
def solve_dp(k):
    size = 26  # 第二维度的大小

    # DP 初始值（递归边界）
    # 一般是一个全为 1 的列向量，对应初始值 f[0][j]=1 或者递归边界 dfs(0,j)=1
    f0 = Matrix.new_matrix(size, 1)
    for i in range(size):
        f0.data[i][0] = 1

    # 例如，递推式中的 f[i][j] += f[i-1][k] * 2，提取系数得 m[j][k] = 2
    m = Matrix.new_matrix(size, size)
    for j in range(size):
        m.data[j][(j + 1) % size] = 3  # 如果 f[i][j] += f[i-1][j+1] * 3
        m.data[j][(j + 2) % size] = 5  # 如果 f[i][j] += f[i-1][j+2] * 5

    # fk 和 f0 一样，都是长为 size 的列向量
    fk = m.pow_mul(k, f0)

    # 现在 fk[i][0] 就是 f[k][i] 或者 dfs(k,i)
    # 特别地，fk[0][0] 就是 f[k][0] 或者 dfs(k,0)
    ans = 0
    for row in fk.data:
        ans += row[0]  # 举例 ans = sum(f[k])
    ans %= mod

    return ans

# -----------------------------------------------------------------------------

# 广义斐波那契数列
# a(n) = p*a(n-1) + q*a(n-2)
# ！数列下标从 1 开始，n 从 1 开始
# https://www.luogu.com.cn/problem/P1349
# https://www.luogu.com.cn/problem/P1939
def calc_fibonacci(p, q, a1, a2, n):
    if n == 1:
        return a1 % mod

    # 变形得到 [f[n], f[n-1]] = [[p, q], [1, 0]] = [f[n-1], f[n-2]]
    # 也可以用打家劫舍的状态机写法理解，其中 f[i][0] 表示 i 可选可不选，f[i][1] 表示 i 一定不能选
    # f[i][0] += p*f[i-1][0] 不选 i
    # f[i][0] += q*f[i-1][1] 选 i，那么 i-1 一定不能选
    # f[i][1] = f[i-1][0]
    # 提取系数得 m[0][0] = p，m[0][1] = q，m[1][0] = 1
    m = Matrix([
        [p, q],
        [1, 0]
    ])
    f2 = Matrix([
        [a2],
        [a1]
    ])
    # 结果是列向量 [f[n], f[n-1]]，取第一项
    fn = m.pow_mul(n - 2, f2)
    return fn.data[0][0]

def pow_mod(a, n, mod):
    """快速幂"""
    res = 1
    while n > 0:
        if n % 2 > 0:
            res = res * a % mod
        a = a * a % mod
        n //= 2
    return res

# NxN 矩阵求逆
# 模板题 https://www.luogu.com.cn/problem/P4783
def matrix_inv(A):
    """矩阵求逆"""
    # 增广一个单位矩阵
    n = len(A)
    m = 2 * n
    a = []
    for i in range(n):
        row = [0] * m
        for j in range(n):
            row[j] = A[i][j]  # or read
        row[n + i] = 1
        a.append(row)

    for i in range(n):
        for j in range(i, n):
            if a[j][i] != 0:
                a[i], a[j] = a[j], a[i]
                break

        if a[i][i] == 0:
            # 矩阵不是满秩的
            return None

        inv = pow_mod(a[i][i], mod - 2, mod)
        for j in range(i, m):
            a[i][j] = a[i][j] * inv % mod

        for j in range(n):
            if j != i:
                inv = a[j][i]
                for k in range(i, m):
                    a[j][k] = (a[j][k] - inv * a[i][k] % mod + mod) % mod

    # 结果保存在 a 右侧
    res = []
    for i in range(n):
        res.append(a[i][n:])
    return res

# 高斯消元 Gaussian elimination O(n^3)   列主元消去法
# 求解 Ax=B，A 为方阵，返回解（无解或有无穷多组解）
# https://en.wikipedia.org/wiki/Gaussian_elimination
# https://en.wikipedia.org/wiki/Pivot_element#Partial_and_complete_pivoting
# https://oi-wiki.org/math/gauss/
# 总结 https://cloud.tencent.com/developer/article/1087352
# https://cp-algorithms.com/linear_algebra/determinant-gauss.html
# https://algs4.cs.princeton.edu/code/edu/princeton/cs/algs4/GaussianElimination.java.html
# https://algs4.cs.princeton.edu/code/edu/princeton/cs/algs4/GaussJordanElimination.java.html
# 模板题 https://www.luogu.com.cn/problem/P3389 https://www.luogu.com.cn/problem/P2455
#       https://codeforces.com/problemset/problem/21/B
# https://www.luogu.com.cn/problem/P4035
# https://www.luogu.com.cn/problem/P6030 与 SCC 结合
#
# 三对角矩阵算法 托马斯算法
# https://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm
# https://codeforces.com/problemset/problem/24/D 2400
def gauss_jordan_elimination(A, B):
    """高斯-约旦消元法"""
    eps = 1e-8
    n = len(A)
    # 构造增广矩阵 (or read)
    a = []
    for i in range(n):
        row = [float(A[i][j]) for j in range(n)]
        row.append(float(B[i]))
        a.append(row)

    row = 0
    for col in range(n):
        # 列主元消去法：减小误差，把正在处理的未知数系数的绝对值最大的式子换到第 row 行
        pivot = row
        for i in range(row, n):
            if abs(a[i][col]) > abs(a[pivot][col]):
                pivot = i

        # 这一列全为 0，表明无解或有无穷多解，具体是哪一种需要消元完成后才知道
        if abs(a[pivot][col]) < eps:
            continue

        a[row], a[pivot] = a[pivot], a[row]

        # 初等行变换：把正在处理的未知数的系数变为 1
        for j in range(col + 1, n + 1):
            a[row][j] /= a[row][col]

        # 消元，构造简化行梯阵式
        for i in range(n):
            if i != row:
                # 用当前行对其余行消元：从第 i 个式子中消去第 col 个未知数
                for j in range(col + 1, n + 1):
                    a[i][j] -= a[i][col] * a[row][j]
        row += 1

    if row < n:
        for r in a[row:]:
            if abs(r[n]) > eps:
                return None, False  # 无解
        return None, True  # 有无穷多解

    res = [a[i][n] for i in range(n)]
    return res, False

# EXTRA: 求行列式（对结果模 mod）
# https://en.wikipedia.org/wiki/Determinant
# 参考 https://www.luogu.com.cn/blog/Stormy-Rey/calculate-det
# https://www.luogu.com.cn/problem/P7112
def determinant(a, mod):
    """求矩阵行列式"""
    n = len(a)
    res, sign = 1, 1
    for i in range(n):
        for j in range(i + 1, n):
            while a[i][i] != 0:
                div = a[j][i] // a[i][i]
                for k in range(i, n):
                    a[j][k] = (a[j][k] - a[i][k] * div % mod + mod) % mod
                a[i], a[j], sign = a[j], a[i], -sign
            a[i], a[j], sign = a[j], a[i], -sign

    for i in range(n):
        res = res * a[i][i] % mod
    res = (res * sign + mod) % mod
    return res

# 求矩阵的特征多项式
# todo https://www.cnblogs.com/ywwyww/p/8522541.html
#  https://www.luogu.com.cn/problem/P7776

# 最短递推式
# todo
#  Berlekamp–Massey 算法
#  https://mzhang2021.github.io/cp-blog/berlekamp-massey/
#  https://en.wikipedia.org/wiki/Berlekamp%E2%80%93Massey_algorithm
#  https://oi-wiki.org/math/berlekamp-massey/
#  https://codeforces.com/blog/entry/61306
#  https://codeforces.com/blog/entry/96199
#  https://www.luogu.com.cn/problem/P5487
#  https://codeforces.com/problemset/problem/1511/F 2700
#  https://codeforces.com/problemset/problem/506/E
#  https://leetcode.cn/problems/total-characters-in-string-after-transformations-ii/description/
#  - https://leetcode.cn/problems/total-characters-in-string-after-transformations-ii/solutions/2973816/os2logtjie-fa-bmsuan-fa-you-hua-ju-zhen-gdknh/

# 线性基（异或空间的极大线性无关子集）
# 可以用来解决「子序列异或和」相关问题
# https://oi-wiki.org/math/basis/
# https://en.wikipedia.org/wiki/Basis_(linear_algebra)
# 【推荐】https://www.luogu.com.cn/blog/Marser/solution-p3812
# 线性基学习笔记 https://oi.men.ci/linear-basis-notes/
# XOR basis without linear algebra https://codeforces.com/blog/entry/100066
# https://www.luogu.com.cn/blog/i207M/xian-xing-ji-xue-xi-bi-ji-xie-ti-bao-gao
# 讲解+题单 https://www.cnblogs.com/UntitledCpp/p/13912602.html
# https://www.luogu.com.cn/blog/Troverld/xian-xing-ji-xue-xi-bi-ji
# todo 讲到了线性基的删除操作 https://blog.csdn.net/a_forever_dream/article/details/83654397
# 线性基求交 https://www.cnblogs.com/BakaCirno/p/11298102.html
# https://zhuanlan.zhihu.com/p/139074556
#
# 模板题 https://www.luogu.com.cn/problem/P3812 https://loj.ac/p/113
# 题单 https://www.luogu.com.cn/training/11251
# https://codeforces.com/problemset/problem/845/G 2300 异或最短路/最长路
# - https://www.luogu.com.cn/problem/P4151
# https://codeforces.com/problemset/problem/1101/G 2300
# https://codeforces.com/problemset/problem/662/A 2400 博弈
# https://codeforces.com/problemset/problem/959/F 2400
# https://codeforces.com/problemset/problem/1163/E 2400
# https://codeforces.com/problemset/problem/1902/F 2400 LCA
# - https://www.luogu.com.cn/problem/P3292 [SCOI2016] 幸运数字
# https://codeforces.com/problemset/problem/1100/F 2500
# https://codeforces.com/problemset/problem/1427/E 2500 构造
# https://codeforces.com/problemset/problem/1778/E 2500
# https://codeforces.com/problemset/problem/724/G 2600 图上线性基
# https://codeforces.com/problemset/problem/251/D 2700 输出具体方案
# - https://atcoder.jp/contests/abc141/tasks/abc141_f 简单版本
# https://codeforces.com/problemset/problem/19/E 2900 图上线性基
# https://codeforces.com/problemset/problem/587/E 2900
# - https://www.luogu.com.cn/problem/P5607
# https://codeforces.com/problemset/problem/1336/E2 3500
# https://atcoder.jp/contests/abc141/tasks/abc141_f
# https://www.luogu.com.cn/problem/P3857
# https://loj.ac/p/2978
# - https://codeforces.com/problemset/problem/895/C

class XorBasis:
    def __init__(self, a=None):
        self.b = [0] * 64  # 核心就这一个，or 32

        self.right_most = [0] * 64  # 注意这里是 0
        self.right_most_zero = -1   # 注意这里是 -1

        self.num = 0
        self.or_val = 0

        self.can_be_zero = False  # 见 min_xor 和 kth_xor
        self.basis = None         # 见 init_once

        if a:
            for v in a:
                self.insert(v)

    # 尝试插入 v，看能否找到一个新的线性无关基
    # 针对稀疏二进制的写法 https://leetcode.cn/problems/partition-array-for-maximum-xor-and-and/solution/shi-zi-bian-xing-xian-xing-ji-pythonjava-3e80/
    def insert(self, v):
        self.or_val |= v
        # 从高到低遍历，保证计算 max_xor 的时候，参与 XOR 的基的最高位（或者说二进制长度）是互不相同的
        for i in range(len(self.b) - 1, -1, -1):
            if v >> i == 0:  # 由于大于 i 的位都被我们异或成了 0，所以 v>>i 的结果只能是 0 或 1
                continue
            if self.b[i] == 0:  # x 和之前的基是线性无关的
                self.b[i] = v   # 新增一个基，最高位为 i
                self.num += 1
                return True
            v ^= self.b[i]  # 保证每个基的二进制长度互不相同
        # 正常循环结束，此时 x=0，说明一开始的 x 可以被已有基表出，不是一个线性无关基
        self.can_be_zero = True  # 说明存在非空集合，异或和为 0
        return False

    # EXTRA: 从高到低，对于二进制长度相同的基，选更靠右的
    # https://atcoder.jp/contests/abc223/tasks/abc223_h
    # https://codeforces.com/problemset/problem/1902/F 2400
    # https://codeforces.com/problemset/problem/1100/F 2500
    # https://codeforces.com/problemset/problem/1778/E 2500
    def insert_right_most(self, idx, v):
        for i in range(len(self.b) - 1, -1, -1):
            if v >> i == 0:
                continue
            if self.b[i] == 0:
                self.b[i] = v
                self.right_most[i] = idx
                self.num += 1
                return True
            # 替换掉之前的基，尽量保证基的下标都是最新的
            # 替换后，可能插入新的基，也可能淘汰掉旧的基
            if idx > self.right_most[i]:
                idx, self.right_most[i] = self.right_most[i], idx
                v, self.b[i] = self.b[i], v
            v ^= self.b[i]
        self.can_be_zero = True  # 没有找到，但这说明了可以选一些数使得异或和为 0
        self.right_most_zero = max(self.right_most_zero, idx)
        return False

    # v 能否被线性基表出
    def decompose(self, v):
        for i in range(len(self.b) - 1, -1, -1):
            if v >> i == 0:
                continue
            # 配合 insert_right_most
            # self.b[i] == 0 or self.right_most[i] < lower_index
            if self.b[i] == 0:
                return False
            v ^= self.b[i]
        return True

    # 返回能被线性基表出的最大值
    # 如果线性基为空，返回 0
    # https://www.luogu.com.cn/problem/P3812
    # https://loj.ac/p/113
    # https://leetcode.cn/problems/partition-array-for-maximum-xor-and-and/solutions/3734850/shi-zi-bian-xing-xian-xing-ji-pythonjava-3e80/
    def max_xor(self):
        res = 0
        for i in range(len(self.b) - 1, -1, -1):
            res = max(res, res ^ self.b[i])
        return res

    def max_xor_with_val(self, val):
        res = val
        for i in range(len(self.b) - 1, -1, -1):
            res = max(res, res ^ self.b[i])
        return res

    # 配合 insert_right_most
    def max_xor_with_lower_index(self, lower_index):
        res = 0
        for i in range(len(self.b) - 1, -1, -1):
            if (res >> i) & 1 == 0 and self.right_most[i] >= lower_index:
                res = max(res, res ^ self.b[i])
        return res

    # 考虑插入的过程，因为每一次跳转操作，x 的二进制最高位必定单调降低，所以不可能插入两个二进制最高位相同的数。
    # 而此时，线性基中最小值异或上其他数，必定会增大。
    # 所以，直接输出线性基中的最小值即可。
    def min_xor(self):
        if self.can_be_zero:
            return 0
        for i in range(len(self.b)):
            if self.b[i] > 0:
                return self.b[i]

    def init_once(self):
        if self.basis is not None:
            return
        tmp = self.b[:]
        self.basis = []
        for i in range(len(tmp)):
            if tmp[i] == 0:
                continue
            for j in range(i - 1, -1, -1):
                if (tmp[i] >> j) & 1 > 0:
                    tmp[i] ^= tmp[j]
            self.basis.append(tmp[i])

    # 线性基能表出的所有不同元素中的第 k 小值（不允许空）
    # k 从 1 开始
    # https://loj.ac/p/114 http://acm.hdu.edu.cn/showproblem.php?pid=3949
    def kth_xor(self, k):
        self.init_once()
        if self.can_be_zero:  # 0 是最小的
            k -= 1  # 占用了一个数
        if k >= 1 << len(self.basis):  # 非空子集有 2^len(self.basis) - 1 个
            return -1
        xor = 0
        for i, v in enumerate(self.basis):
            if (k >> i) & 1 > 0:
                xor ^= v
        return xor

    # todo https://www.luogu.com.cn/problem/P4869
    def rank(self, xor):
        raise NotImplementedError("todo")

    # https://codeforces.com/problemset/problem/1902/F
    def merge(self, other):
        for i in range(len(other.b) - 1, -1, -1):
            v = other.b[i]
            if v > 0:
                self.insert(v)

""" 矩阵树定理 基尔霍夫定理 Kirchhoff's theorem
https://oi-wiki.org/graph/matrix-tree/
https://en.wikipedia.org/wiki/Kirchhoff%27s_theorem

https://atcoder.jp/contests/jsc2021/tasks/jsc2021_g
https://atcoder.jp/contests/abc253/tasks/abc253_h
https://atcoder.jp/contests/abc323/tasks/abc323_g
"""

# 线性规划（单纯形法）  LP, linear programming (simplex method)
# https://en.wikipedia.org/wiki/Mathematical_optimization
# https://en.wikipedia.org/wiki/Linear_programming
# https://en.wikipedia.org/wiki/Integer_programming
# https://en.wikipedia.org/wiki/Simplex_algorithm
# todo https://oi-wiki.org/math/simplex/
#      https://zhuanlan.zhihu.com/p/31644892
#  https://algs4.cs.princeton.edu/code/edu/princeton/cs/algs4/LinearProgramming.java.html
#
# todo https://uoj.ac/problem/179
#  https://codeforces.com/problemset/problem/1430/G https://codeforces.com/blog/entry/83614?#comment-709868
#  https://codeforces.com/problemset/problem/375/E
#  NOI08 志愿者招募 https://www.luogu.com.cn/problem/P3980
#       整数线性规划与全幺模矩阵 https://www.acwing.com/file_system/file/content/whole/index/content/2197334/