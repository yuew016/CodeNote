# 1.6 Depth First Search

碰到让你找**所有**方案的题，一定是DFS

90%DFS的题，要么是排列，要么是组合

• 什么时候用 DFS? • 求所有方案时

• 怎么解决DFS? • 不是排列就是组合

• 复杂度怎么算? • O\(答案个数 \* 构造每个答案的时间复杂度\)

• 非递归怎么办? • 必“背”程序

## 4.1 组合搜索 Combination

问题模型:求出所有满足条件的“组合”。 判断条件:组合中的元素是顺序无关的。 时间复杂度:与 2^n 相关。

一般来说，如果面试官不特别要求的话，DFS都可以使用递归\(Recursion\)的方式来实现。 递归三要素是实现递归的重要步骤

### 子集Subsets

中文English

给定一个含不同整数的集合，返回其所有的子集

```python
class Solution:
    #1. 递归的定义
    #以 S 开头的，配上 nums 以 index 开始的数所有组合放到 results 里
    def search(self, nums, S, index):
      #3. 递归的出口
        if index == len(nums):
            self.results.append(list(S)) #deep copy
            return
        #2. 递归的拆解 (如何进入下一层)
        #选了 nums[index]
        S.append(nums[index])
        self.search(nums, S, index + 1)
        #不选 nums[index]
        S.pop()
        self.search(nums, S, index + 1)

    def subsets(self, nums):
        self.results = []
        self.search(sorted(nums), [], 0)
        return self.results
```

### 数字组合

给定一个候选数字的集合 `candidates` 和一个目标值 `target`. 找到 `candidates` 中所有的和为 `target` 的组合.

在同一个组合中, `candidates` 中的某个数字不限次数地出现.

与 Subsets 比较

* Combination Sum 限制了组合中的数之和 • 加入一个新的参数来限制
* Subsets 无重复元素，Combination Sum 有重复元素 • 需要先去重
* Subsets 一个数只能选一次，Combination Sum 一个数可以选很多次

  • 搜索时从 index 开始而不是从 index + 1

```python
class Solution:
    """
    @param candidates: A list of integers
    @param target: An integer
    @return: A list of lists of integers 84%
    """
    def combinationSum(self, candidates, target):
        # write your code here
        candidates = sorted(list(set(candidates))) #去重
        results = [] 
        self.dfs(candidates, target, 0, [], results) 
        return results

    def dfs(self, candidates, target, start, combination, results):
        # 递归的出口：target <= 0
        if target < 0:
            return

        if target == 0:
            # deepcooy
            return results.append(list(combination))

        # 递归的拆解：挑一个数放到 combination 里
        for i in range(start, len(candidates)):
            # [2] => [2,2]
            combination.append(candidates[i])
            self.dfs(candidates, target - candidates[i], i, combination, results)
            # [2,2] => [2]
            combination.pop()  # backtracking
```

复杂度分析：

* 时间复杂度：$O\( n^\(target/min\) \)$, $n$为集合中数字个数，$min$为集合中最小的数字
* 每个位置可以取集合中的任意数字，最多有$target/min$个数字。
* 空间复杂度：$O\( n^\(target/min\) \)$, $n$为集合中数字个数，$min$为集合中最小的数字
* 对于用来保存答案的列表，最多有$n^\(target/min\)$种组合

### 数字组合 II

中文English

给定一个数组 `num` 和一个整数 `target`. 找到 `num` 中所有的数字之和为 `target` 的组合.

解释: 解集不能包含重复的组合

1. 在同一个组合中, `num` 中的每一个数字仅能被使用一次

```python
class Solution:
    """
    @param num: Given the candidate numbers
    @param target: Given the target number
    @return: All the combinations that sum to target
    """
    def combinationSum2(self, num, target):
        # write your code here
        result = []
        if not num or len(num) == 0:
            return result
        self.dfs(sorted(num), target, 0, [], result)
        return result

    def dfs(self, num, target, index, combination, result):
        if target < 0:
            return
        if target == 0:
            return result.append(list(combination))
        for i in range(index, len(num)):
            if i!=index and num[i] == num[i-1]:
                continue
            combination.append(num[i])
            self.dfs(num, target - num[i], i + 1, combination, result)
            combination.pop()
```

### Palindrome Partitioning

使用 append + pop 的方式

```python
class Solution:
    """
    @param: s: A string
    @return: A list of lists of string
    """
    def partition(self, s):
        # write your code here
        results = []
        if not s or len(s) == 0:
            return results
        self.dfs(s, [], results)
        return results

    def dfs(self, s, string, results):
        if len(s) == 0:
            return results.append(list(string))
        for i in range(1, len(s)+1):
            now = s[:i]
            if self.is_palindrome(now):
                string.append(now)
                self.dfs(s[i:], string, results)
                string.pop()

    def is_palindrome(self, s):
        return s == s[::-1]
```

有什么可以优化的地方? 判断回文串：dp

```python
def get_is_palindrome(self, s):
        n = len(s)
        is_palindrome = [[False] * n for _ in range(n)]
        for i in range(n):
            is_palindrome[i][i] = True
        for i in range(n - 1):
            is_palindrome[i][i + 1] = (s[i] == s[i + 1])

        for delta in range(2, n):
            for i in range(n - delta):
                j = i + delta
                is_palindrome[i][j] = is_palindrome[i + 1][j - 1] and s[i] == s[j]

        return is_palindrome
```

## 4.2 排列搜索 Permutation

问题模型:求出所有满足条件的“排列”。

判断条件:组合中的元素是顺序“相关”的。

时间复杂度:与 n! 相关。

### permutation

给定一个数字列表，返回其所有可能的排列。\(无重复数字\)

```python
class Solution:
    """
    @param: nums: A list of integers.
    @return: A list of permutations.
    """
    def permute(self, nums):
        results = []

        # 如果数组为空直接返回空
        if nums is None:
            return []

        # dfs
        used = [0] * len(nums)
        self.dfs(nums, used, [], results)
        return results

    def dfs(self, nums, used, current, results):

        # 找到一组排列，已到达边界条件
        if len(nums) == len(current):
            # 因为地址传递，在最终回溯后current为空导致results中均为空列表
            # 所以不能写成results.append(current)
            results.append(list(current))
            return

        for i in range(len(nums)):
            # i位置这个元素已经被用过
            if used[i]:
                continue
            # 继续递归
            current.append(nums[i])
            used[i] = 1
            self.dfs(nums, used, current, results)
            used[i] = 0
            current.pop()
```

### Permutations II

和没有重复元素的 Permutation 一题相比，只加了两句话：

1. Arrays.sort\(nums\) // 排序这样所有重复的数
2. if \(i &gt; 0 && nums[i](https://www.jiuzhang.com/solution/permutations-ii/) == nums[i - 1](https://www.jiuzhang.com/solution/permutations-ii/) && !visited[i - 1](https://www.jiuzhang.com/solution/permutations-ii/)\) { continue; } // 跳过会造成重复的情况

   ```python
   class Solution:
       """
       @param: :  A list of integers
       @return: A list of unique permutations
       """

       def permuteUnique(self, nums):
           # write your code here
           if not nums or len(nums) == 0:
               return [[]]

           nums = sorted(nums)
           results = []
           used = [0]*len(nums)
           self.dfs(nums, [], results, used)
           return results

       def dfs(self, nums, cur, results,used):
           if len(cur) == len(nums):
               results.append(list(cur))
               return
           for i in range(len(nums)):
               if used[i]:
                   continue
               elif i!=0 and nums[i] == nums[i-1] and used[i-1] == 0:
                   continue
               cur.append(nums[i])
               used[i] = 1 
               self.dfs(nums, cur, results,used)
               cur.pop()
               used[i] = 0
   ```

### N Queens

用 visited 来标记 列号，横纵坐标之和，横纵坐标之差 有没有被用过

```python
class Solution:
    """
    @param: n: The number of queens
    @return: All distinct solutions
    """
    def solveNQueens(self, n):
        # write your code here
        board = []
        visited = {
            'col':set(),
            'sum': set(),
            'diff': set(),
            }
        self.search(n, [], visited, board)
        return board

    def search(self, n, plan, visited, board):

        if len(plan) == n:
            board.append(self.draw(plan))
            return

        raw = len(plan)
        for col in range(n):
            if not self.is_valid(col,plan, visited):
                continue
            plan.append(col)
            visited['col'].add(col)
            visited['sum'].add(raw + col)
            visited['diff'].add(raw - col)
            self.search(n, plan, visited, board)
            visited['col'].remove(col)
            visited['sum'].remove(raw + col)
            visited['diff'].remove(raw - col)
            plan.pop()

    def is_valid(self, col, plan, visited):
        row = len(plan)
        if col in visited['col']:
            return False
        if row + col in visited['sum']:
            return False
        if row - col in visited['diff']:
            return False
        return True

    def draw(self, plan):
        board = []
        n = len(plan)
        for col in plan:
            row_string = ''.join(['Q' if c == col else '.' for c in range(n)])
            board.append(row_string)
        return board
```

### 搜索，动态规划，二叉树的时间复杂度计算通用公式

搜索的时间复杂度：O\(答案总数  _构造每个答案的时间\)\` 举例：Subsets问题，求所有的子集。子集个数一共 2^n，每个集合的平均长度是 O\(n\) 的，所以时间复杂度为 O\(n_  2^n\)，同理 Permutations 问题的时间复杂度为：O\(n \* n!\)

动态规划的时间复杂度：O\(状态总数 \* 计算每个状态的时间复杂度\)\` 举例：triangle，数字三角形的最短路径，状态总数约 O\(n^2\) 个，计算每个状态的时间复杂度为 O\(1\)——就是求一下 min。所以总的时间复杂度为 O\(n^2\)

用分治法解决二叉树问题的时间复杂度：O\(二叉树节点个数 \* 每个节点的计算时间\)\` 举例：二叉树最大深度。二叉树节点个数为 N，每个节点上的计算时间为 O\(1\)。总的时间复杂度为 O\(N\)

## 4.3 必“背”程序

Tree Traversal

[http://www.jiuzhang.com/solutions/binary-tree-preorder-traversal/](http://www.jiuzhang.com/solutions/binary-tree-preorder-traversal/)

[http://www.jiuzhang.com/solutions/binary-tree-inorder-traversal/](http://www.jiuzhang.com/solutions/binary-tree-inorder-traversal/)

[http://www.jiuzhang.com/solutions/binary-tree-postorder-traversal/](http://www.jiuzhang.com/solutions/binary-tree-postorder-traversal/)

[http://www.jiuzhang.com/solutions/binary-search-tree-iterator/](http://www.jiuzhang.com/solutions/binary-search-tree-iterator/)

Combination

[http://www.jiuzhang.com/solutions/subsets/](http://www.jiuzhang.com/solutions/subsets/)

Permutation

[http://www.jiuzhang.com/solutions/permutations/](http://www.jiuzhang.com/solutions/permutations/)

