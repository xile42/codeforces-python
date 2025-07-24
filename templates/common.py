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


"""
codeforces-python: 算法竞赛Python3模板库
#2: 回文字符串生成器(记忆化搜索)
https://github.com/xile42/codeforces-python/blob/main/templates/common.py
"""
class PalindromeStringGenerator:

    def __init__(self, s: str) -> None:

        self.s = s
        self.n = len(s)
        self.palindromes = set()
        self._init_palindromes()

    def _init_palindromes(self) -> None:

        # 生成所有单字符回文
        for c in self.s:
            self.palindromes.add(c)

        # 以下搜索似乎只能分为两步, 如果记忆化搜索参数带当前字符串, 会退化为O(n! * 2 ^ n)
        # 预计算所有可能的状态
        self._precompute_states()

        # 根据状态回溯生成字符串
        self._generate_palindromes()

        # 排序结果
        self.palindromes = sorted(self.palindromes, key=lambda x: len(x), reverse=True)

    def _precompute_states(self) -> None:
        """ 记忆化搜索所有合法状态 """

        # 奇回文
        for i in range(self.n):
            self._dfs(i, i, 1 << i)

        # 偶回文
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if self.s[i] == self.s[j]:
                    self._dfs(i, j, (1 << i) | (1 << j))  # 注意, 保证状态大小关系以缩减状态

    @cache
    def _dfs(self, left: int, right: int, vis: int) -> bool:
        """ 是否能形成有效回文 """

        is_valid = False

        for i in range(self.n):
            if vis & (1 << i):
                continue
            for j in range(self.n):
                if not (vis & (1 << j)) and i != j and self.s[i] == self.s[j]:
                    ni, nj = min(i, j), max(i, j)
                    if self._dfs(ni, nj, vis | (1 << i) | (1 << j)):
                        is_valid = True

        return is_valid or (left == right) or (self.s[left] == self.s[right])

    def _generate_palindromes(self) -> None:
        """ 根据有效状态生成回文字符串 """

        # 需要维护一个从状态到字符串集合的映射
        state_map = defaultdict(set)

        # 初始状态（单字符）
        for i in range(self.n):
            state_map[(i, i, 1 << i)].add(self.s[i])

        # 双字符状态
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if self.s[i] == self.s[j]:
                    state_map[(i, j, (1 << i) | (1 << j))].add(self.s[i] + self.s[j])  # 注意, 保证状态大小关系以缩减状态

        # 动态扩展状态
        for _ in range(self.n // 2):
            new_map = defaultdict(set)
            for (l, r, vis), strs in state_map.items():
                for i in range(self.n):
                    if vis & (1 << i):
                        continue
                    for j in range(self.n):
                        if not (vis & (1 << j)) and i != j and self.s[i] == self.s[j]:
                            ni, nj = min(i, j), max(i, j)  # 保持有序, 减少记忆化搜索状态, 注意不能直接交换i, j, 否则下一循环下标错误
                            new_vis = vis | (1 << i) | (1 << j)
                            for s in strs:
                                new_str = self.s[i] + s + self.s[j]
                                new_map[(ni, nj, new_vis)].add(new_str)
            state_map.update(new_map)

        # 收集所有回文
        for strs in state_map.values():
            self.palindromes.update(strs)

class PalindromeStringGenerator:

    def __init__(self, s: str) -> None:

        self.s = s
        self.n = len(s)
        self.palindromes = set()
        self._init_palindromes()

    def _init_palindromes(self) -> None:

        # 生成所有单字符回文
        for c in self.s:
            self.palindromes.add(c)

        # 奇回文(单中心)
        for i in range(self.n):
            self._dfs(i, i, 1 << i, self.s[i])

        # 偶回文(双中心)
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if self.s[i] == self.s[j]:
                    self._dfs(i, j, (1 << i) | (1 << j), self.s[i] + self.s[j])

        self.palindromes = sorted(self.palindromes, key=lambda x: len(x), reverse=True)

    @cache
    def _dfs(self, left: int, right: int, vis: int, current: str) -> None:

        self.palindromes.add(current)

        for i in range(self.n):
            if vis & (1 << i):
                continue
            for j in range(self.n):
                if not (vis & (1 << j)) and i != j and self.s[i] == self.s[j]:
                    new_vis = vis | (1 << i) | (1 << j)
                    ni, nj = min(i, j), max(i, j)  # 保持有序, 减少记忆化搜索状态, 注意不能直接交换i, j, 否则下一循环下标错误
                    next_s = self.s[i] + current + self.s[j]
                    self._dfs(ni, nj, new_vis, next_s)

class PalindromeStringGenerator:

    def __init__(self, s: str) -> None:

        self.s = s
        self.palindromes = set()
        self._init_palindromes()

    def _init_palindromes(self) -> None:

        n = len(self.s)
        for mask in range(1, 1 << n):  # 跳过空子集(mask = 0)
            subset = [self.s[i] for i in range(n) if mask & (1 << i)]
            self._process_subset(subset)

        self.palindromes = sorted(self.palindromes, key=lambda x: len(x), reverse=True)

    def _process_subset(self, subset: List[str]) -> None:

        counter = Counter(subset)
        odd_chars = [char for char, freq in counter.items() if freq % 2 == 1]

        if len(odd_chars) > 1:  # 无法形成回文
            return

        middle = odd_chars[0] if odd_chars else ""
        left_chars = list()
        for char, freq in counter.items():
            left_chars.extend([char] * (freq // 2))

        # 生成左半部分的全排列
        for perm in set(permutations(left_chars)):
            left = "".join(perm)
            self.palindromes.add(left + middle + left[::-1])

class PalindromeStringGenerator:

    def __init__(self, s: str) -> None:
        self.s = s
        self.n = len(s)
        self.palindromes: Set[str] = set()
        self._generate()

    def _generate(self) -> None:
        # 生成所有单字符回文
        for c in self.s:
            self.palindromes.add(c)

        # 使用动态规划同时计算状态和字符串
        self.dp = defaultdict(set)

        # 初始化基础状态
        for i in range(self.n):
            self._add_state(i, i, 1 << i, self.s[i])
            for j in range(i + 1, self.n):
                if self.s[i] == self.s[j]:
                    self._add_state(i, j, (1 << i) | (1 << j), self.s[i] + self.s[j])

        # 动态扩展（确保使用已验证的合法状态）
        for _ in range(self.n // 2):
            new_dp = defaultdict(set)
            for (l, r, vis), strs in self.dp.items():
                for i in range(self.n):
                    if vis & (1 << i):
                        continue
                    for j in range(self.n):
                        if not (vis & (1 << j)) and i != j and self.s[i] == self.s[j]:
                            ni, nj = min(i, j), max(i, j)
                            new_vis = vis | (1 << i) | (1 << j)
                            for s in strs:
                                new_str = self.s[i] + s + self.s[j]
                                new_dp[(ni, nj, new_vis)].add(new_str)
            self._update_dp(new_dp)

        # 排序结果
        self.palindromes = sorted(self.palindromes, key=lambda x: (-len(x), x))

    def _add_state(self, l: int, r: int, vis: int, s: str) -> None:
        """添加状态并验证有效性"""
        if self._is_valid_state(l, r, vis):
            self.dp[(l, r, vis)].add(s)
            self.palindromes.add(s)

    @cache
    def _is_valid_state(self, l: int, r: int, vis: int) -> bool:
        """验证状态是否可形成回文"""
        if l == r or self.s[l] == self.s[r]:
            return True

        for i in range(self.n):
            if vis & (1 << i):
                continue
            for j in range(self.n):
                if not (vis & (1 << j)) and i != j and self.s[i] == self.s[j]:
                    ni, nj = min(i, j), max(i, j)
                    if self._is_valid_state(ni, nj, vis | (1 << i) | (1 << j)):
                        return True
        return False

    def _update_dp(self, new_dp: dict) -> None:
        """更新状态并去重"""
        for state, strs in new_dp.items():
            if self._is_valid_state(*state):
                self.dp[state].update(strs)
                self.palindromes.update(strs)
