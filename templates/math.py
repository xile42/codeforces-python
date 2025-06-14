from typing import *


"""
[埃氏筛/埃拉托斯特尼筛法(Sieve of Eratosthenes)]
用于快速筛选质数，可选同时记录最小质因数或所有质因数

[时间复杂度]
    O(NloglogN)  # 近似线性
[空间复杂度]
    O(N)

[相关链接]
    1. https://oi-wiki.org/math/number-theory/sieve/
    2. https://en.wikipedia.org/wiki/Sieve_of_Eratosthenes
"""


"""
codeforces-python: 算法竞赛Python3模板库
#1: 基础埃氏筛
https://github.com/xile42/codeforces-python/blob/main/templates/math.py
"""
class EratosthenesSieve:

    def __init__(self, mx: int=10 ** 5 + 2, need_factors: bool=False) -> None:

        self.is_prime = [True] * mx
        self.is_prime[0] = self.is_prime[1] = False  # 忽略了越界检查

        if need_factors:
            self.factors = [[] for _ in range(mx)]

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
