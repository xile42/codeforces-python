from typing import *
from functools import cache
from collections import defaultdict


"""
codeforces-python: 算法竞赛Python3模板库
#1: 回文数生成器(<= 指定值)
https://github.com/xile42/codeforces-python/blob/main/templates/common.py
"""
class PalindromeNumberGenerator:

    def __init__(self, mx: int = 10 ** 9) -> None:

        self.mx = mx
        self.pal = list()
        self._init_palindromes()

    def _init_palindromes(self) -> None:

        base = 1
        while True:

            # 生成奇数长度回文数
            for i in range(base, base * 10):
                x = i
                t = i // 10
                while t > 0:
                    x = x * 10 + t % 10
                    t = t // 10
                if x > self.mx:
                    break
                self.pal.append(x)

            # 生成偶数长度回文数
            else:
                for i in range(base, base * 10):
                    x = i
                    t = i
                    while t > 0:
                        x = x * 10 + t % 10
                        t = t // 10
                    if x > self.mx:
                        break
                    self.pal.append(x)
                else:
                    base *= 10
                    continue

            break
