from typing import *
from sortedcontainers import SortedList


"""
[前缀树/字典树/单词查找树(Trie)]
适用于多串前缀/后缀匹配
另类解读: 如果将字符串长度视作定值 L 的话, trie 树是一种 O(nL) 排序，O(L) 查询的数据结构

[时间复杂度]
    add: O(N)
    find: O(N)
    remove: O(N)
    rank: O(N)
    get_kth: O(N)
    count_prefix_of_s: O(N)
    count_s_has_prefix_p: O(N)
    count_distinct_substring: O(N^2)

[空间复杂度]
    O(N*L*C)  # N为字符串数量，L为平均长度，C为字符集大小(实际空间取决于共享前缀情况，最坏情况下为O(N*L*C))
"""


"""
CodeForces题单: (by 灵茶山艾府 https://github.com/EndlessCheng/codeforces-go/tree/master )

    [一般]
    1. https://codeforces.com/problemset/problem/271/D 1800 满足一些条件的本质不同子串个数
    2. https://codeforces.com/problemset/problem/455/B 1900
    3. https://codeforces.com/problemset/problem/113/B 2000 满足一些条件的本质不同子串个数
    4. https://codeforces.com/problemset/problem/514/C 2000
    5. https://codeforces.com/problemset/problem/965/E 2200 与贪心堆结合
    6. https://codeforces.com/problemset/problem/557/E 2300
"""


"""
codeforces-python: 算法竞赛Python3模板库
#1: 前缀树/字典树/单词查找树
https://github.com/xile42/codeforces-python/blob/main/templates/trie.py
"""
class TrieNode:

    __slots__ = ("son", "cnt", "sum", "val", "sorted_list")

    def __init__(self) -> None:
        self.son: Dict[Hashable, TrieNode] = dict()  # 子节点
        self.cnt: int = 0  # 当前节点对应的完整字符串的个数(以当前节点为结尾的完整字符串个数)
        self.sum: int = 0  # 子树 cnt 之和(有多少个完整字符串，包含以当前节点为起点的后缀字符串; 特别的，root的sum值相当于Trie中的字符串数量)
        self.val: int = 0  # 根据需要额外存储的信息
        self.sorted_list: SortedList = SortedList()  # 所有经过该节点的字符串 s 的排序

    def empty(self) -> bool:
        return len(self.son) == 0  # 是否叶子节点


class Trie:

    def __init__(self, need_sorted_list: bool = False) -> None:
        self.root = TrieNode()
        self.need_sorted_list = need_sorted_list

    _OFFSET = ord("a")

    @staticmethod
    def ord(c: str) -> int:
        return ord(c) - Trie._OFFSET

    @staticmethod
    def chr(v: int) -> str:
        return chr(v + Trie._OFFSET)

    def add(self, s: str, val: int = 0) -> TrieNode:
        """插入字符串 s，附带值 val，返回插入后字符串末尾对应的节点"""
        node = self.root
        for c in s:
            node.sum += 1  # 子树 cnt 之和(截止到node的字符串，对应的字符串是多少个完整字符串的前缀)
            idx = self.ord(c)
            if idx not in node.son:
                node.son[idx] = TrieNode()
            node = node.son[idx]
            if self.need_sorted_list:
                node.sorted_list.add(s)
        node.cnt += 1  # o 对应的完整字符串的个数
        node.val = val

        return node

    def find(self, s: str) -> Tuple[TrieNode, bool]:
        """查找字符串 s 与字典树中字符串的最长公共前缀，返回最后一个匹配的节点(最长公共前缀)，以及是否找到 s"""
        node = self.root
        for c in s:
            idx = self.ord(c)
            if idx not in node.son:
                return node, False
            node = node.son[idx]

        return node, node.cnt != 0

    def remove(self, s: str) -> Optional[TrieNode]:
        """删除字符串 s，返回字符串末尾对应的节点"""
        # 检查 s 是否在 trie 中
        node = self.root
        for c in s:
            idx = self.ord(c)
            if idx not in node.son:
                return None  # s 不在 trie 中
            node = node.son[idx]

        parents: List[TrieNode] = list()
        node = self.root
        for c in s:
            node.sum -= 1
            parents.append(node)
            idx = self.ord(c)
            node = node.son[idx]

        node.cnt -= 1
        if node.cnt == 0:
            for i in range(len(s) - 1, -1, -1):
                parent_node = parents[i]
                idx = self.ord(s[i])
                del parent_node.son[idx]  # 完全删除节点
                if not parent_node.empty():
                    break

        return node

    def rank(self, s: str) -> int:
        """求小于 s 的字符串个数, 相当于从 0 开始的 rank 值"""
        ans = 0
        node = self.root
        for c in s:
            idx = self.ord(c)
            # 累加在 idx 之前的子树大小
            for i in range(idx):
                if i in node.son:
                    ans += node.son[i].sum  # 当前字符小于 c 的所有字符串均合法
            if idx not in node.son:
                return ans
            node = node.son[idx]  # 当前字符等于 c 的情况，要进一步看后续
            ans += node.cnt  # 以 c 结尾的字符串个数

        # return ans  # 小于等于 s 的字符串数目
        return ans - node.cnt  # 小于 s 的字符串数目

    def get_kth(self, k: int) -> str:
        """求第 k 小(k 从 0 开始)，需要保证 trie 中至少有 k+1 个字符串"""
        s = list()
        node = self.root
        while True:
            for idx in sorted(node.son.keys()):
                son = node.son[idx]
                # 子树 son 中的字符串都比答案小
                if k >= son.sum:
                    k -= son.sum
                    continue
                s.append(self.chr(idx))
                k -= son.cnt
                if k < 0:
                    return "".join(s)
                node = son
                break
            else:
                raise ValueError("k is too large")

    def count_prefix_of_s(self, s: str) -> int:
        """返回字符串 s 在 trie 中的前缀个数，即有多少完整字符串是s的前缀"""
        cnt = 0
        node = self.root
        for c in s:
            idx = self.ord(c)
            if idx not in node.son:
                return cnt
            node = node.son[idx]
            cnt += node.cnt

        return cnt

    def count_s_has_prefix_p(self, p: str) -> int:
        """返回 trie 中前缀包含 p 的字符串个数"""
        node = self.root
        for c in p:
            idx = self.ord(c)
            if idx not in node.son:
                return 0
            node = node.son[idx]

        return node.sum

    def count_distinct_substring(self, s: str) -> int:
        """s 的本质不同子串数量 O(n^2), 本质不同字串指: 字符串中所有不同的子串(去重)"""
        cnt = 0
        for i in range(len(s)):
            node = self.root
            for c in s[i:]:
                idx = self.ord(c)
                if idx not in node.son:
                    node.son[idx] = TrieNode()
                    cnt += 1
                node = node.son[idx]

        return cnt
