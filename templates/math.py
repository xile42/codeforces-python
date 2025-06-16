from typing import *


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
