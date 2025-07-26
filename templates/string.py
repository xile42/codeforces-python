import random
from typing import *


"""
[单模-字符串哈希]
基于多项式滚动哈希实现, 使用大质数避免冲突

[时间复杂度]
    预处理: O(N)
    查询子串哈希: O(1)
[空间复杂度]
    O(N)

[相关链接]
    1. https://oi-wiki.org/string/hash/
    2. https://cp-algorithms.com/string/string-hashing.html
"""


"""
codeforces-python: 算法竞赛Python3模板库
#1: 单模-字符串哈希
https://github.com/xile42/codeforces-python/blob/main/templates/string.py
"""
class StringHashSingleMod:

    def __init__(self, s: str):

        self.s = s
        self.n = len(s)
        # mod 与 base 随机一个即可, 以避免被hack
        self.mod = 1070777777
        self.base = 900000000 - random.randint(0, 100000000)
        self.pow_base = [1] * (self.n + 1)
        self.pre_hash = [0] * (self.n + 1)
        self._run()

    def _run(self):
        """ 预处理哈希值, 多项式系数前缀和数组 """

        for i in range(1, self.n + 1):
            self.pow_base[i] = (self.pow_base[i - 1] * self.base) % self.mod
            self.pre_hash[i] = (self.pre_hash[i - 1] * self.base + ord(self.s[i - 1])) % self.mod

    def get_hash(self, l: int, r: int) -> int:
        """ 获取子串哈希值 [l, r] """

        return ((self.pre_hash[r + 1] - self.pre_hash[l] * self.pow_base[r + 1 - l]) % self.mod + self.mod) % self.mod


"""
[双模-字符串哈希]
双模哈希降低冲突概率，使用两个不同的大质数

[时间复杂度]
    预处理: O(N)
    查询子串哈希: O(1) 
[空间复杂度]
    O(N)

[相关链接]
    1. https://codeforces.com/blog/entry/60445
"""


"""
codeforces-python: 算法竞赛Python3模板库
#2: 双模-字符串哈希
https://github.com/xile42/codeforces-python/blob/main/templates/string.py
"""
class StringHashDoubleMod:

    def __init__(self, s: str):

        self.s = s
        self.n = len(s)
        # mod 与 base 随机一组即可, 以避免被hack
        self.mod1 = 1070777777
        self.mod2 = 1000000007
        self.base1 = 900000000 - random.randint(0, 100000000)
        self.base2 = 900000000 - random.randint(0, 100000000)
        self.pow_base = [(1, 1)] * (self.n + 1)
        self.pre_hash = [(0, 0)] * (self.n + 1)
        self._run()

    def _run(self):
        """ 预处理哈希值, 多项式系数前缀和数组 """

        for i in range(1, self.n + 1):
            pb1, pb2 = self.pow_base[i - 1]
            self.pow_base[i] = (pb1 * self.base1 % self.mod1, pb2 * self.base2 % self.mod2)
            ph1, ph2 = self.pre_hash[i - 1]
            self.pre_hash[i] = (
                (ph1 * self.base1 + ord(self.s[i - 1])) % self.mod1,
                (ph2 * self.base2 + ord(self.s[i - 1])) % self.mod2
            )

    def get_hash(self, l: int, r: int) -> Tuple[int, int]:
        """ 获取子串双模哈希值 [l, r] """

        h1 = ((self.pre_hash[r + 1][0] - self.pre_hash[l][0] * self.pow_base[r + 1 - l][0]) % self.mod1 + self.mod1) % self.mod1
        h2 = ((self.pre_hash[r + 1][1] - self.pre_hash[l][1] * self.pow_base[r + 1 - l][1]) % self.mod2 + self.mod2) % self.mod2

        return (h1, h2)


"""
[KMP算法(Knuth-Morris-Pratt Algorithm)]
字符串匹配算法
定义: π[i] 表示模式串 P[0..i] 的最长公共真前后缀长度
  - 真前缀：不包含末尾字符的子串 (如 "abc"的真前缀有 "a", "ab", 即不能取到最后一个元素)
  - 真后缀：不包含首字符的子串 (如 "abc"的真后缀有 "c", "bc", 即不能取到最后一个元素)

[时间复杂度]
    预处理: O(M)  # M为模式串长度
    匹配: O(N)  # N为文本串长度
[空间复杂度]
    O(M)

[相关链接]
    1. https://oi-wiki.org/string/kmp/
    2. https://cp-algorithms.com/string/prefix-function.html
"""


"""
codeforces-python: 算法竞赛Python3模板库
#3: KMP算法
https://github.com/xile42/codeforces-python/blob/main/templates/string.py
"""
class KMP:

    def __init__(self, pattern: str):

        self.pattern = pattern
        self.n = len(self.pattern)
        self.pi = self._run()

    def _run(self) -> List[int]:
        """ 计算前缀函数 """
        
        pi = [0] * self.n
        match = 0
        for i in range(1, self.n):
            while match > 0 and self.pattern[i] != self.pattern[match]:
                match = pi[match - 1]
            if self.pattern[i] == self.pattern[match]:
                match += 1
            pi[i] = match

        return pi

    def search(self, text: str) -> List[int]:
        """ 在文本中搜索模式串 """
        
        pos = list()
        match = 0
        for i in range(len(text)):
            while match > 0 and text[i] != self.pattern[match]:
                match = self.pi[match - 1]
            if text[i] == self.pattern[match]:
                match += 1
            if match == len(self.pattern):
                pos.append(i - len(self.pattern) + 1)
                match = self.pi[match - 1]

        return pos

    def has_repeated_substring(self) -> bool:
        """ 检测模式串是否由某个子串重复多次构成 """

        if self.n == 0:
            return False
        # 检查是否满足 n % (n - π[n-1]) == 0 且 π[n-1] != 0
        return self.pi[-1] != 0 and self.n % (self.n - self.pi[-1]) == 0

    def get_repeated_substring(self) -> Optional[str]:
        """ 获取构成模式串的循环节(如果存在) """

        if self.n == 0 or self.pi[-1] == 0:
            return None

        if self.n % (self.n - self.pi[-1]) == 0:
            return self.pattern[:self.n - self.pi[-1]]

        return None


"""
[Z算法/扩展KMP]
计算与字符串本身每个后缀的最长公共前缀

[时间复杂度]
    预处理: O(N)
    模式匹配: O(N+M)
[空间复杂度]
    O(N)

[相关链接]
    1. https://cp-algorithms.com/string/z-function.html
    2. https://oi-wiki.org/string/z-func/
"""


"""
codeforces-python: 算法竞赛Python3模板库
#4: Z算法/扩展KMP
https://github.com/xile42/codeforces-python/blob/main/templates/string.py
"""
class ZAlgorithm:

    def __init__(self, s: str):
        self.s = s
        self.z = self._compute_z()

    def _compute_z(self) -> List[int]:
        """计算Z数组"""
        n = len(self.s)
        z = [0] * n
        box_l, box_r = 0, 0
        for i in range(1, n):
            if i <= box_r:
                z[i] = min(z[i - box_l], box_r - i + 1)
            while i + z[i] < n and self.s[z[i]] == self.s[i + z[i]]:
                z[i] += 1
                box_l, box_r = i, i + z[i]
        z[0] = n
        return z

    def search(self, pattern: str) -> List[int]:
        """在文本中搜索模式串"""
        s = pattern + '#' + self.s
        z = self._compute_z()
        pos = []
        for i, l in enumerate(z[len(pattern) + 1:]):
            if l == len(pattern):
                pos.append(i)
        return pos


"""
[马拉车算法(Manacher)]
线性时间求最长回文子串

[时间复杂度]
    预处理: O(N)
    查询最长回文: O(1)
[空间复杂度]
    O(N)

[相关链接]
    1. https://cp-algorithms.com/string/manacher.html
    2. https://oi-wiki.org/string/manacher/
"""


"""
codeforces-python: 算法竞赛Python3模板库
#5: Manacher
https://github.com/xile42/codeforces-python/blob/main/templates/string.py
"""
class Manacher:

    def __init__(self, s: str):
        self.s = s
        self.t = self._preprocess()
        self.half_len, self.box_params = self._compute()

    def _preprocess(self) -> str:
        """预处理字符串"""
        t = ['^']
        for c in self.s:
            t.extend(['#', c])
        t.extend(['#', '$'])
        return ''.join(t)

    def _compute(self) -> Tuple[List[int], Tuple[int, int]]:
        """计算回文信息"""
        n = len(self.t)
        half_len = [0] * n
        box_m, box_r = 0, 0

        for i in range(1, n - 1):
            if i < box_r:
                i_mirror = 2 * box_m - i
                half_len[i] = min(half_len[i_mirror], box_r - i)

            while self.t[i - half_len[i]] == self.t[i + half_len[i]]:
                half_len[i] += 1
                if i - half_len[i] < 0 or i + half_len[i] >= n:
                    break

            if i + half_len[i] > box_r:
                box_m, box_r = i, i + half_len[i]

        return half_len, (box_m, box_r)

    def longest_palindrome(self) -> str:
        """获取最长回文子串"""
        max_len = max(self.half_len)
        center = self.half_len.index(max_len)
        start = (center - max_len) // 2
        end = start + max_len - 1
        return self.s[start:end + 1]


"""
[最小表示法]
求循环同构字符串中字典序最小的表示

[时间复杂度]
    O(N)
[空间复杂度]
    O(N)

[相关链接]
    1. https://oi-wiki.org/string/minimal-string/
"""


"""
codeforces-python: 算法竞赛Python3模板库
#6: 最小表示法
https://github.com/xile42/codeforces-python/blob/main/templates/string.py
"""
class SmallestRepresentation:

    def __init__(self, s: str):
        self.s = s
        self.result = self._compute()

    def _compute(self) -> str:
        """计算最小表示"""
        n = len(self.s)
        s = self.s + self.s
        i, j = 0, 1
        while j < n:
            k = 0
            while k < n and s[i + k] == s[j + k]:
                k += 1
            if k >= n:
                break
            if s[i + k] > s[j + k]:
                i = max(i + k + 1, j)
                j = i + 1
            else:
                j += k + 1
        return s[i:i + n]

    def get(self) -> str:
        """获取最小表示结果"""
        return self.result


"""
[子序列自动机]
快速判断字符串是否为另一个字符串的子序列

[时间复杂度]
    预处理: O(N*|Σ|)  # |Σ|为字符集大小
    单次查询: O(M)    # M为模式串长度
[空间复杂度]
    O(N*|Σ|)

[相关链接]
    1. https://oi-wiki.org/string/seq-automaton/
"""


"""
codeforces-python: 算法竞赛Python3模板库
#7: 子序列自动机
https://github.com/xile42/codeforces-python/blob/main/templates/string.py
"""
class SubsequenceAutomaton:

    def __init__(self, s: str):
        self.s = s
        self.n = len(s)
        self.nxt = self._build_automaton()

    def _build_automaton(self) -> List[List[int]]:
        """构建自动机"""
        nxt = [[self.n] * 26 for _ in range(self.n + 1)]
        for i in range(self.n - 1, -1, -1):
            nxt[i] = nxt[i + 1].copy()
            nxt[i][ord(self.s[i]) - ord('a')] = i
        return nxt

    def is_subsequence(self, t: str) -> bool:
        """判断是否是子序列"""
        i = -1
        for c in t:
            i = self.nxt[i + 1][ord(c) - ord('a')]
            if i == self.n:
                return False
        return True


"""
[后缀数组(Suffix Array)]
后缀排序及相关应用，包含height数组(最长公共前缀)

[时间复杂度]
    构建: O(NlogN)  # 使用倍增算法
    LCP查询: O(1)   # 需要RMQ预处理
[空间复杂度]
    O(N)

[相关链接]
    1. https://oi-wiki.org/string/sa/
    2. https://cp-algorithms.com/string/suffix-array.html
"""


"""
codeforces-python: 算法竞赛Python3模板库
#8: 后缀数组
https://github.com/xile42/codeforces-python/blob/main/templates/string.py
"""
class SuffixArray:

    def __init__(self, s: str):
        self.s = s
        self.sa, self.rank, self.height = self._compute_suffix_array()

    def _compute_suffix_array(self) -> Tuple[List[int], List[int], List[int]]:
        """计算后缀数组"""
        n = len(self.s)
        sa = list(range(n))
        rank = [ord(c) for c in self.s]
        k = 1
        while k < n:
            sa.sort(key=lambda i: (rank[i], rank[i + k] if i + k < n else -1))
            tmp = [0] * n
            tmp[sa[0]] = 0
            for i in range(1, n):
                tmp[sa[i]] = tmp[sa[i - 1]]
                if (rank[sa[i]] != rank[sa[i - 1]] or
                        (sa[i] + k < n and sa[i - 1] + k < n and
                         rank[sa[i] + k] != rank[sa[i - 1] + k])):
                    tmp[sa[i]] += 1
            rank = tmp
            k *= 2

        height = [0] * n
        h = 0
        for i in range(n):
            if rank[i] == 0:
                continue
            j = sa[rank[i] - 1]
            while i + h < n and j + h < n and self.s[i + h] == self.s[j + h]:
                h += 1
            height[rank[i]] = h
            if h > 0:
                h -= 1

        return sa, rank, height

    def longest_common_prefix(self, i: int, j: int) -> int:
        """计算两个后缀的最长公共前缀"""
        if i == j:
            return len(self.s) - i
        ri, rj = self.rank[i], self.rank[j]
        if ri > rj:
            ri, rj = rj, ri
        return min(self.height[ri + 1:rj + 1])


"""
[LCP数组]
预处理任意两个后缀的最长公共前缀

[时间复杂度]
    预处理: O(N²)
    单次查询: O(1)
[空间复杂度]
    O(N²)

[相关链接]
    1. https://cp-algorithms.com/string/suffix-array.html#number-of-different-substrings
"""


"""
codeforces-python: 算法竞赛Python3模板库
#9: LCP数组
https://github.com/xile42/codeforces-python/blob/main/templates/string.py
"""
class LCPArray:

    def __init__(self, s: str):
        self.s = s
        self.lcp = self._compute_lcp()

    def _compute_lcp(self) -> List[List[int]]:
        """计算LCP数组"""
        n = len(self.s)
        lcp = [[0] * (n + 1) for _ in range(n + 1)]
        for i in range(n - 1, -1, -1):
            for j in range(n - 1, -1, -1):
                if self.s[i] == self.s[j]:
                    lcp[i][j] = lcp[i + 1][j + 1] + 1
        return lcp

    def get_lcp(self, i: int, j: int) -> int:
        """获取两个位置的最长公共前缀长度"""
        return self.lcp[i][j]
