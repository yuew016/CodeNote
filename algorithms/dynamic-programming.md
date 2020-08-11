# 1.6 Dynamic Programming

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

