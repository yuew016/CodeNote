# 1.7 Dynamic Programming

## 通过一道经典题理解动态规划

 • 递归与动规的联系与区别: 有没有重复计算  
 • 记忆化搜索的本质:动态规划 

动态规划与分治的区别? 重复计算!

### Triangle

解决方法: • DFS: Traverse • DFS: Divide Conquer • Divide Conquer + Memorization • Traditional Dynamic Programming

```python
#DFS: Traverse 
#time: O(2^n), n = height
class Solution:
    def minimumTotal(self, triangle: List[List[int]]) -> int:
        self.best = sys.maxsize
        self.traverse(triangle,0,0,0)
        return self.best
    
    def traverse(self, triangle, x,y,sum):
        if x == len(triangle):
            if sum < self.best:
                self.best = sum
            return
        self.traverse(triangle,x+1, y, sum+triangle[x][y])
        self.traverse(triangle, x+1, y+1, sum+triangle[x][y])


#DFS: Divide Conquer
#time:O(2^n)
class Solution:
    def minimumTotal(self, triangle: List[List[int]]) -> int:
        def divideConquer(x,y):
            if x == len(triangle):
                return 0
            return triangle[x][y] + min(divideConquer(x+1, y), divideConquer(x+1, y+1))
        
        return divideConquer(0,0)
```

dp:

```python
#dp:记忆化搜索 Divide Conquer + Memorization
#时间复杂度 O(n2) 空间复杂度 O(n2)
class Solution:
    def minimumTotal(self, triangle: List[List[int]]) -> int:
        def divideConquer(x,y,memo):
            if x == len(triangle)-1:
                return triangle[x][y]
            if (x,y) in memo:
                return memo[(x,y)]
            memo[(x,y)] = triangle[x][y] + min(divideConquer(x+1, y, memo), divideConquer(x+1, y+1, memo))
            return memo[(x,y)]
        
        return divideConquer(0,0,{})
```

```python
#dp:多重循环，自顶向下
#state: dp[i][j] 代表从 i,j  走到最顶层的最短路径值
#时间复杂度 O(n2) 空间复杂度 O(n2)
class Solution:
    def minimumTotal(self, triangle: List[List[int]]) -> int:
        n = len(triangle)
        dp = [[0]*(i+1) for i in range(n)]
        
        #initialize: 最顶层，两边
        dp[0][0] = triangle[0][0]
        for i in range(1,n):
            dp[i][0], dp[i][i] = triangle[i][0]+dp[i-1][0], triangle[i][i]+dp[i-1][i-1]
            
        #dp[i][j] = triangle[i][j] + min(dp[i-1][j], dp[i-1][j-1])
        for i in range(1, n):
            for j in range(1,i):
                dp[i][j] = triangle[i][j] + min(dp[i-1][j], dp[i-1][j-1])
        return min(dp[n-1])
```

dp优化空间：滚动数组 i%2

自底向上的动态规划，多重循环实现 使用滚动数组进行空间优化

时间复杂度 O\(n2\)O\(n2\) 空间复杂度 O\(n\)O\(n\) （额外空间耗费）

```python
class Solution:
    """
    @param triangle: a list of lists of integers
    @return: An integer, minimum path sum
    """
    def minimumTotal(self, triangle):
        n = len(triangle)
        
        # state: dp[i][j] 代表从 0, 0 走到 i, j 的最短路径值
        dp = [[0] * n, [0] * n]
        
        # initialize: 初始化起点
        dp[0][0] = triangle[0][0]
        
        # function: dp[i][j] = min(dp[i - 1][j - 1], dp[i - 1][j]) + triangle[i][j]
        # i, j 这个位置是从位置 i - 1, j 或者 i - 1, j - 1 走过来的
        for i in range(1, n):
            dp[i % 2][0] = dp[(i - 1) % 2][0] + triangle[i][0]
            dp[i % 2][i] = dp[(i - 1) % 2][i - 1] + triangle[i][i]
            for j in range(1, i):
                dp[i % 2][j] = min(dp[(i - 1) % 2][j], dp[(i - 1) % 2][j - 1]) + triangle[i][j]
               
        # answer: 最后一层的任意位置都可以是路径的终点
        return min(dp[(n - 1) % 2])
```

```python
class Solution:
    """
    @param triangle: a list of lists of integers
    @return: An integer, minimum path sum
    """
    def minimumTotal(self, triangle):
        # write your code here
        #自底向上的方式的多重循环动态规划, 滚动数组优化空间
        #time:n^2 space:n 
        #1.state: dp[i][j] = min sum bottom to [i][j]
        n = len(triangle)
        dp = [[0]*n, [0]*n]
        
        #2.initialize: dp[n-1]
        for i in range(n):
            dp[(n-1)%2][i] = triangle[n-1][i]
        
        #3. func:dp[i%2][j] = min(dp[(i+1)%2][j], dp[(i+1)%2][j+1]) + triangle[i][j]
        for i in range(n-2, -1, -1):
            for j in range(i+1):
                dp[i%2][j] = min(dp[(i+1)%2][j], dp[(i+1)%2][j+1]) + triangle[i][j]
        #ans: dp[0][0]      
        return dp[0%2][0]
 
        
        

```

## dp两种方法

多重循环 vs 记忆化搜索

多重循环：

优点:正规，大多数面试官可以 接受，存在空间优化可能性。

缺点:思考有难度。

记忆化搜索：

优点:容易从搜索算法直接转化过来。有的时候可以节省更多的时间。

缺点:递归

## 什么时候使用动态规划

• 适用动态规划的三个条件:

满足下面三个条件之一: 极有可能  
• 求最大值最小值  
• 判断是否可行  
• 统计方案个数

• 不适用动态规划的三个条件：  
1. 求出**所有 具体 的方案**而非方案 **个数** :dfs  
[http://www.lintcode.com/problem/palindrome-partitioning/ ](http://www.lintcode.com/problem/palindrome-partitioning/%20)  
2. 输入数据是一个 **集合** 而不是 **序列:** dp不能换位置，必须有方向性  
[http://www.lintcode.com/problem/longest-consecutive-sequence/](http://www.lintcode.com/problem/longest-consecutive-sequence/)  
3. 暴力算法的复杂度已经是多项式级别  
 • 动态规划擅长与优化指数级别复杂度\(2^n,n!\)到多项式级别复杂度\(n^2,n^3\)   
• 不擅长优化n^3到n^2

• 则 极不可能 使用动态规划求解

## 动规四要素

状态 State  
 • 灵感，创󿲕力，存储小规模问题的结果 •   
方程 Function   
• 状态之间的联系，怎么通过小的状态，来算大的状态 •   
初始化 Initialization   
• 最极限的小状态是什么, 起点 •   
答案 Answer   
• 最大的那个状态是什么，终点

动态规划只能记录一种最优的方案 动规无法记录所有的最优方案

## 面试中常见动态规划的分类 

**• 坐标\(矩阵\)动态规划  
 • 接龙型动态规划**

  
****• **state**:  
 • f\[x\] 表示我从起点走到坐标x......  
 • f\[x\]\[y\] 表示我从起点走到坐标x,y......  
****• **function**: 研究走到x,y这个点之前的一步   
• **initialize**: 起点  
 • **answer**: 终点

初始化一个二维的动态规划时 就去**初始化第0行和第0列**

## Minimum Path Sum



follow up:



## Climbing Stairs

follow up:



## 跳跃游戏 I && II

 [http://www.lintcode.com/en/problem/jump-game/](http://www.lintcode.com/en/problem/jump-game/) [http://www.lintcode.com/en/problem/jump-game-ii/](http://www.lintcode.com/en/problem/jump-game-ii/) 

这个题最优的方法是使用“贪心法”，动态规划复杂度较高

## Unique Paths

4个向下，4个向右 = C\(8,4\)

Follow up:



**接龙型动态规划:**

## Longest Increasing Subsequence

## Russian Doll Envelopes

## Largest Divisible Subset

## 青蛙过河

http://www.lintcode.com/en/problem/frog-jump/

## [Decode Ways](https://leetcode.com/problems/decode-ways/)

```python
class Solution:
    def numDecodings(self, s: str) -> int:
        if not s:
            return 0

        dp = [0 for x in range(len(s) + 1)] 

        # base case initialization
        dp[0] = 1 
        dp[1] = 0 if s[0] == "0" else 1   #(1)

        for i in range(2, len(s) + 1): 
            # One step jump
            if 0 < int(s[i-1:i]) <= 9:    #(2)
                dp[i] += dp[i - 1]
            # Two step jump
            if 10 <= int(s[i-2:i]) <= 26: #(3)
                dp[i] += dp[i - 2]
        return dp[len(s)] 
        
```

## Maximum sum such that no two elements are adjacent

```python
'''
Loop for all elements in arr[] and maintain two sums incl and excl where:
incl[i] = Max sum including the i element 
excl[i] = Max sum excluding the i element.

excl[i] = max(incl[i-1] , excl[i-1]) 
incl[i] = excl[i-1] + arr[i]
At the end of the loop return max of incl and excl
'''
#code
def maxSum(arr, n):
    if n == 1:
        return arr[0]
    include = [0]*n
    exclude = [0]*n
    include[0] = arr[0]
    for i in range(1,n):
        include[i] = arr[i] + exclude[i-1]
        exclude[i] = max(include[i-1], exclude[i-1])
    return max(include[n-1], exclude[n-1])

if __name__ == '__main__':
    T = int(input())

    for _ in range(T):
        n=int(input())
        arr = [int(x) for x in input().split()]

        print(maxSum(arr,n))
```

