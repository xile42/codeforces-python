from typing import *
from operator import *
from functools import reduce


"""
[埃氏筛/埃拉托斯特尼筛法(Sieve of Eratosthenes)]
用于快速筛选质数, 可选同时记录所有质因数
虽然时间复杂度不是严格线性, 但缓存友好, 在 <= 1e7 的场景更快(py 1s内)
质因数分解:
    1. 可通过need_factors=True, 预处理所有质因数, 但空间复杂度较高
    2. 也可通过need_lpf=True, 预处理每个数的最小质因数, 以O(logN)质因数分解, 空间复杂度更低

[时间复杂度]
    initialize: O(NloglogN)  # 近似线性, 常数小
    factorize with lpf: O(logN)  # 每个数仅存储lpf, 空间复杂度更优
[空间复杂度]
    sieve: O(N)

[相关链接]
    1. https://oi-wiki.org/math/number-theory/sieve/
    2. https://en.wikipedia.org/wiki/Sieve_of_Eratosthenes
"""


"""
codeforces-python: 算法竞赛Python3模板库
#1: 埃氏筛
https://github.com/xile42/codeforces-python/blob/main/templates/math.py
"""
class EratosthenesSieve:

    def __init__(self, mx: int=10 ** 5 + 2, need_factors: bool=False, need_lpf: bool=False) -> None:

        self.is_prime = [True] * mx
        self.is_prime[0] = self.is_prime[1] = False  # 忽略了越界检查

        if need_factors:
            self.factors = [[] for _ in range(mx)]  # 无重复质因数列表

        if need_lpf:
            # lpf[x] = 0  x <= 1, 无意义
            # lpf[x] = x  x 是质数
            # lpf[x] = y  x 是合数, 则 y 是 x 的最小质因数
            self.lpf = [0] * mx

        for i in range(2, mx):

            if not self.is_prime[i]:
                continue

            if need_factors:
                self.factors[i].append(i)

            if need_lpf:
                self.lpf[i] = i  # 质数的最小质因子是自身

            for j in range((i + i) if need_factors else (i * i), mx, i):  # 仅筛质数可以从i * i开始, 求质因数只能从i + i开始; 前者更快但不影响时间复杂度
                self.is_prime[j] = False
                if need_factors:
                    self.factors[j].append(i)
                if need_lpf and self.lpf[j] == 0:
                    self.lpf[j] = i  # 记录最小质因子

    def factorize(self, x: int) -> Dict[int, int]:

        assert hasattr(self, 'lpf'), "LPF array not initialized. Set need_lpf=True in constructor."

        if x < 2:
            return dict()

        factors = dict()
        while x > 1:
            lpf = self.lpf[x]
            factors[lpf] = factors.get(lpf, 0) + 1
            x //= lpf

        return factors


"""
[欧拉筛/线性筛法(Sieve of Euler)]
用于严格线性时间筛选质数, 可选同时记录所有质因数
时间复杂度严格线性, 但分支较多, 访存不连续, 在 >= 1e7 的场景更快
质因数分解:
    1. 可通过need_factors=True, 预处理所有质因数, 但空间复杂度较高
    2. 也可通过need_lpf=True, 预处理每个数的最小质因数, 以O(logN)质因数分解, 空间复杂度更低

[时间复杂度]
    initialize: O(N)  # 严格线性
    factorize with lpf: O(logN)  # 每个数仅存储lpf, 空间复杂度更优
[空间复杂度]
    sieve: O(N)

[相关链接]
    1. https://oi-wiki.org/math/number-theory/sieve/
    2. https://en.wikipedia.org/wiki/Sieve_of_Eratosthenes#Euler's_sieve
"""


"""
codeforces-python: 算法竞赛Python3模板库
#2: 欧拉筛
https://github.com/xile42/codeforces-python/blob/main/templates/math.py
"""
class EulerSieve:

    def __init__(self, mx: int=10 ** 5 + 2, need_factors: bool=False, need_lpf: bool=False) -> None:

        self.is_prime = [True] * mx
        self.is_prime[0] = self.is_prime[1] = False  # 忽略了越界检查
        self.primes = list()  # 所有质数

        if need_factors:
            self.factors = [[] for _ in range(mx)]  # 有重复质因数列表

        if need_lpf:
            # lpf[x] = 0  x <= 1, 无意义
            # lpf[x] = x  x 是质数
            # lpf[x] = y  x 是合数, 则 y 是 x 的最小质因数
            self.lpf = [0] * mx

        for i in range(2, mx):

            if self.is_prime[i]:
                self.primes.append(i)
                if need_factors:
                    self.factors[i].append(i)
                if need_lpf:
                    self.lpf[i] = i

            for p in self.primes:

                if i * p >= mx:
                    break

                self.is_prime[i * p] = False

                if need_factors:
                    self.factors[i * p] = self.factors[i].copy()  # 复制i的质因数
                    self.factors[i * p].append(p)  # 添加当前质数p

                if need_lpf:
                    if self.lpf[i * p] == 0:
                        self.lpf[i * p] = p  # 最小质因子

                if i % p == 0:  # 保证每个合数只被最小质因数筛一次
                    if need_factors and i != p:  # 如果i是p的倍数且不是p本身，需要修正质因数列表
                        self.factors[i * p] = self.factors[i].copy()
                    break

    def factorize(self, x: int) -> Dict[int, int]:

        assert hasattr(self, 'lpf'), "LPF array not initialized. Set need_lpf=True in constructor."

        if x < 2:
            return dict()

        factors = dict()
        while x > 1:
            lpf = self.lpf[x]
            factors[lpf] = factors.get(lpf, 0) + 1
            x //= lpf

        return factors


"""
[阶乘与组合数预处理(Factorial, Combination, Permutation, Catalan...)]
用于快速计算阶乘、阶乘逆元、组合数、排列数、卡特兰数等，支持取模

[时间复杂度]
    initialize: O(N)  # 近似线性, 常数小
    query: O(1)  # 每个数仅存储lpf, 空间复杂度更优
[空间复杂度]
    initialize: O(N)

[相关链接]
    1. https://oi-wiki.org/math/combinatorics/basic/
    2. https://en.wikipedia.org/wiki/Factorial
"""


"""
codeforces-python: 算法竞赛Python3模板库
#3: 阶乘与组合数预处理
https://github.com/xile42/codeforces-python/blob/main/templates/math.py
"""
class Factorial:

    def __init__(self, n: int=100_000, mod: int=1_000_000_007) -> None:

        n += 1
        self.mod = mod
        self.f = [1] * n  # 阶乘
        self.g = [1] * n  # 阶乘逆元

        for i in range(1, n):
            self.f[i] = self.f[i - 1] * i % self.mod

        self.g[-1] = pow(self.f[-1], mod - 2, mod)
        for i in range(n - 2, -1, -1):
            self.g[i] = self.g[i + 1] * (i + 1) % self.mod

    def factorial(self, n: int) -> int:  # 阶乘

        return self.f[n]

    def factorial_inverse(self, n: int) -> int:  # 逆元

        return self.g[n]

    def comb(self, n: int, m: int) -> int:  # 带模组合数 C(n, m)

        if n < m or m < 0 or n < 0:  # 不合法情况
            return 0

        return self.f[n] * self.g[m] % self.mod * self.g[n - m] % self.mod

    def perm(self, n: int, m: int) -> int:  # 带模全排列 A(n, m)

        if n < m or m < 0 or n < 0:
            return 0

        return self.f[n] * self.g[n - m] % self.mod

    def catalan(self, n: int) -> int:  # 卡特兰数

        return (self.comb(2 * n, n) - self.comb(2 * n, n - 1)) % self.mod

    def inv(self, n: int) -> int:  # n的逆元 n ^ (-1)

        return self.f[n - 1] * self.g[n] % self.mod


"""
[扩展欧几里得算法(Extended Euclidean Algorithm)]
用于求解贝祖等式: a * x + b * y = gcd(a,b), 同时可以用于求解模逆元
g, x, y = exgcd(a, b)
其中:
    g = gcd(a, b)
    x, y 为 x * a + y * b = gcd(a, b) 的一个解
若 b ≠ 0, 则 |x| <= b, |y| <= a

[时间复杂度]
    O(log min(a,b))  # 与欧几里得算法相同

[相关链接]
    1. https://oi-wiki.org/math/number-theory/gcd/
    2. https://en.wikipedia.org/wiki/Extended_Euclidean_algorithm
"""


"""
codeforces-python: 算法竞赛Python3模板库
#4: 扩展欧几里得算法
https://github.com/xile42/codeforces-python/blob/main/templates/math.py
"""
class ExGCD:

    @staticmethod
    def run(a: int, b: int) -> Tuple[int, int, int]:

        if b == 0:
            return a, 1, 0

        g, x, y = ExGCD.run(b, a % b)

        return g, y, x - (a // b) * y


"""
[中国剩余定理(Chinese Remainder Theorem)]
用于求解一元线性同余方程组，要求模数两两互质
求解一元线性同余方程组, 其中a1, a2, ..., ak两两互质
    { x ≡ r1 (mod a1)
    { x ≡ r2 (mod a2)
    ...
    { x ≡ rk (mod ak)

[时间复杂度]
    O(klogM)  # k是方程数量，M是模数的乘积

[相关链接]
    1. https://oi-wiki.org/math/number-theory/crt/
    2. https://en.wikipedia.org/wiki/Chinese_remainder_theorem
"""


"""
codeforces-python: 算法竞赛Python3模板库
#5: 中国剩余定理
https://github.com/xile42/codeforces-python/blob/main/templates/math.py
"""
class CRT:

    @staticmethod
    def run(k: int, a: List[int], r: List[int]) -> int:

        ans = 0
        n = reduce(mul, a)
        for i in range(k):
            m = n // a[i]
            g, b, y = ExGCD.run(m, a[i])
            assert g == 1, "模数不互质, 无法使用中国剩余定理"
            ans = (ans + r[i] * m * b % n) % n

        return (ans % n + n) % n


"""
[模逆元求解(Modular Multiplicative Inverse)]
用于求解 a 在模 m 下的乘法逆元
a * x ≡ 1 (mod m)
逆元存在的条件: gcd(a, m) = 1 (a, m互质)
扩展欧几里得可以找到x, y满足:
    a * x + b * y = gcd(a, b)
代入a, m得:
    a * x + m * y = gcd(a, m)
gcd(a, m) = 1, 且两边取模m:
    a * x ≡ 1 (mod m)
即系数x即为a在模m下的逆元, 其中x可能为负数,需加m到正数

[时间复杂度]
    O(log a)  # 使用扩展欧几里得算法

[相关链接]
    1. https://oi-wiki.org/math/number-theory/inverse/
    2. https://en.wikipedia.org/wiki/Modular_multiplicative_inverse
"""


"""
codeforces-python: 算法竞赛Python3模板库
#6: 模逆元求解
https://github.com/xile42/codeforces-python/blob/main/templates/math.py
"""
class ModInverse:

    @staticmethod
    def run(a: int, m: int) -> int:

        g, x, y = ExGCD.run(a, m)
        assert g == 1, "逆元不存在, a, m不互质"

        return x % m
