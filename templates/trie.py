from typing import *


"""
[前缀树/字典树/单词查找树(Trie)]
适用于多串前缀/后缀匹配
另类解读：如果将字符串长度视作定值 L 的话，trie 树是一种 O(nL) 排序，O(L) 查询的数据结构

[时间复杂度]
    add: O(L)  # L为字符串长度
    find: O(L)  # L为字符串长度
    remove: O(L)  # L为字符串长度
    rank: O(L)  # L为字符串长度
    kth: O(L)  # L为字符串长度，假设字符集大小为常数
    countPrefixOfString: O(L)  # L为字符串长度
    countStringHasPrefix: O(P)  # P为前缀长度
    countDistinctSubstring: O(N^2)  # N为字符串长度
    maxTrieNode: O(N)  # N为字符串长度
    put_persistent: O(L)  # L为字符串长度

[空间复杂度]
    O(N*L*C)  # N为字符串数量，L为平均长度，C为字符集大小(实际空间取决于共享前缀情况，最坏情况下为O(N*L*C))
"""


class TrieNode:
    __slots__ = ("son", "cnt", "sum", "val")

    def __init__(self) -> None:
        self.son: Dict[Hashable, TrieNode] = dict()  # 子节点
        self.cnt: int = 0  # 当前节点对应的完整字符串的个数(以当前节点为结尾的完整字符串个数)
        self.sum: int = 0  # 子树 cnt 之和(有多少个完整字符串，包含以当前节点为起点的后缀字符串)
        self.val: int = 0  # 根据需要额外存储的信息

    def empty(self) -> bool:

        return not any(self.son.values())  # 是否叶子节点


class Trie:

    def __init__(self) -> None:
        self.root = TrieNode()

    @staticmethod
    def ord(c: str) -> int:
        return ord(c) - ord('a')

    @staticmethod
    def chr(v: int) -> str:
        return chr(v + ord('a'))

    def add(self, s: str, val: int = 0) -> TrieNode:
        """插入字符串 s，附带值 val，返回插入后字符串末尾对应的节点"""
        node = self.root
        for c in s:
            idx = self.ord(c)
            if idx not in node.son:
                node.son[idx] = TrieNode()
            node = node.son[idx]
            node.sum += 1  # 子树 cnt 之和(o 对应的字符串是多少个完整字符串的前缀)
        node.cnt += 1  # o 对应的完整字符串的个数
        node.val = val

        return node

    def dfs(self) -> None:
        """字典树 DFS（模板）"""

        def f(o: TrieNode, sum_: int) -> None:
            if o is None:
                return
            # 统计从 root 到 o 的路径
            sum_ += o.cnt
            # ... 这里可以添加处理逻辑
            for child in o.son.values():
                f(child, sum_)

        f(self.root, 0)

    def find(self, s: str) -> (TrieNode, bool):
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
        parents: List[TrieNode] = list()
        node = self.root
        for c in s:
            parents.append(node)
            idx = self.ord(c)
            if idx not in node.son:
                return None  # s不在trie中
            node = node.son[idx]

        # 确保s在trie中再更新sum值和cnt值

        node.cnt -= 1
        if node.cnt == 0:
            for i in range(len(s) - 1, -1, -1):
                f = parents[i]
                idx = self.ord(s[i])
                del f.son[idx]  # 完全删除节点
                if not f.empty():
                    break

        return node

    def rank(self, s: str) -> int:
        """求小于 s 的字符串个数"""
        k = 0
        o = self.root
        for c in s:
            idx = self.ord(c)
            # 累加在 idx 之前的子树大小
            for i in range(idx):
                if i in o.son:
                    k += o.son[i].sum
            if idx not in o.son:
                return k
            o = o.son[idx]
            k += o.cnt  # 以 c 结尾的字符串个数
        # 上面算的是小于等于 s 的字符串个数
        # 要算小于 s 的字符串个数，要把恰好等于 s 的字符串个数减掉
        return k - o.cnt

    def kth(self, k: int) -> str:
        """求第 k 小（k 从 0 开始）
        需要保证 trie 中至少有 k+1 个字符串"""
        s = []
        o = self.root
        while True:
            for idx in sorted(o.son.keys()):
                son = o.son[idx]
                # 子树 son 中的字符串都比答案小
                if k >= son.sum:
                    k -= son.sum
                    continue
                s.append(self.chr(idx))
                k -= son.cnt
                if k < 0:
                    return ''.join(s)
                o = son
                break
            else:
                raise ValueError("k is too large")

    def countPrefixOfString(self, s: str) -> int:
        """返回字符串 s 在 trie 中的前缀个数"""
        cnt = 0
        o = self.root
        for c in s:
            idx = self.ord(c)
            if idx not in o.son:
                return cnt
            o = o.son[idx]
            cnt += o.cnt
        return cnt

    def countStringHasPrefix(self, p: str) -> int:
        """返回 trie 中前缀为 p 的字符串个数"""
        o = self.root
        for c in p:
            idx = self.ord(c)
            if idx not in o.son:
                return 0
            o = o.son[idx]
        return o.cnt

    def countDistinctSubstring(self, s: str) -> int:
        """s 的本质不同子串数量 O(n^2)"""
        cnt = 0
        for i in range(len(s)):
            o = self.root
            for c in s[i:]:
                idx = self.ord(c)
                if idx not in o.son:
                    o.son[idx] = TrieNode()
                    cnt += 1
                o = o.son[idx]
        return cnt

    @staticmethod
    def maxTrieNode(n: int, tar: str) -> str:
        """构造长为 n 的字符串 s，让 https://codeforces.com/problemset/problem/114/D 用到的 node 个数尽量多"""
        s = []
        cnt = 0
        tar_ord = ord(tar) - ord('a')
        for ch in range(26):
            if ch == tar_ord:
                continue
            m = (n + 1) // 26  # 最后一组末尾没有字母，我们假设有字母，那么虚拟的长度是 n+1
            if cnt < (n + 1) % 26:
                m += 1
            cnt += 1
            s.extend([tar] * (m - 1))
            s.append(chr(ch + ord('a')))
        s.extend([tar] * (n - len(s)))  # 最后一组
        return ''.join(s)

    # 可持久化字典树
    # 注意为了拷贝一份 trieNode，这里的接收器不是指针
    # https://oi-wiki.org/ds/persistent-trie/
    # roots = [TrieNode()]
    # roots.append(roots[-1].put(s))
    def put_persistent(self, s: str) -> 'TrieNode':
        """可持久化字典树的插入操作"""
        if not s:
            self.cnt += 1
            return self
        b = ord(s[0]) - ord('a')
        if b not in self.son:
            self.son[b] = TrieNode()
        self.son[b] = self.son[b].put_persistent(s[1:])
        return self

