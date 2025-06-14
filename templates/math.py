from typing import *


"""
[埃氏筛/埃拉托斯特尼筛法(Sieve of Eratosthenes)]
用于快速筛选质数, 可选同时记录所有质因数
虽然时间复杂度不是严格线性, 但缓存友好, 在 <= 1e7 的场景更快(py 1s内)

[时间复杂度]
    O(NloglogN)  # 近似线性, 常数小
[空间复杂度]
    O(N)

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

    def __init__(self, mx: int=10 ** 5 + 2, need_factors: bool=False) -> None:

        self.is_prime = [True] * mx
        self.is_prime[0] = self.is_prime[1] = False  # 忽略了越界检查

        if need_factors:
            self.factors = [[] for _ in range(mx)]  # 无重复质因数列表

        for i in range(2, mx):

            if not self.is_prime[i]:
                continue

            if need_factors:
                self.factors[i].append(i)

            for j in range((i + i) if need_factors else (i * i), mx, i):  # 仅筛质数可以从i * i开始, 求质因数只能从i + i开始; 前者更快但不影响时间复杂度
                self.is_prime[j] = False
                if need_factors:
                    self.factors[j].append(i)

    def prime_list(self) -> List[int]:

        return [i for i in range(len(self.is_prime)) if self.is_prime[i]]


"""
[欧拉筛/线性筛法(Sieve of Euler)]
用于严格线性时间筛选质数, 可选同时记录所有质因数
时间复杂度严格线性, 但分支较多, 访存不连续, 在 >= 1e7 的场景更快

[时间复杂度]
    O(N)  # 严格线性
[空间复杂度]
    O(N)

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

    def __init__(self, mx: int=10 ** 5 + 2, need_factors: bool=False) -> None:

        self.is_prime = [True] * mx
        self.is_prime[0] = self.is_prime[1] = False  # 忽略了越界检查

        self.primes = []  # 所有质数

        if need_factors:
            self.factors = [[] for _ in range(mx)]

        for i in range(2, mx):
            if self.is_prime[i]:
                self.primes.append(i)
                if need_factors:
                    self.factors[i].append(i)

            for p in self.primes:
                if i * p >= mx:
                    break
                self.is_prime[i * p] = False
                if need_factors:
                    self.factors[i * p] = self.factors[i].copy()  # 复制i的质因数
                    self.factors[i * p].append(p)  # 添加当前质数p

                if i % p == 0:  # 保证每个合数只被最小质因数筛一次
                    if need_factors and i != p:  # 如果i是p的倍数且不是p本身，需要修正质因数列表
                        self.factors[i * p] = self.factors[i].copy()
                    break
