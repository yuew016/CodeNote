# Math

## **A common method to rotate the image**

### **48.** Rotate Image

```python
/*
 * clockwise rotate
 * first reverse up to down, then swap the symmetry 
 * 1 2 3     7 8 9     7 4 1
 * 4 5 6  => 4 5 6  => 8 5 2
 * 7 8 9     1 2 3     9 6 3
*/
class Solution:
    def rotate(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        matrix.reverse()
        for i in range(len(matrix)):
            for j in range(i):
                matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]

#first swap the symmatry, second reverse left and right
class Solution:
    def rotate(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        for i in range(len(matrix)):
            for j in range(i):
                matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
        for row in matrix:
            for j in range(len(row)//2):
                row[j], row[~j] = row[~j], row[j]
                '''
                ~ and -:
                a = [1, 2, 3, 4]
                a[~0] = 4 // a[-1] = 4
                '''
                
            


/*
 * anticlockwise rotate
 * first reverse left to right, then swap the symmetry
 * 1 2 3     3 2 1     3 6 9
 * 4 5 6  => 6 5 4  => 2 5 8
 * 7 8 9     9 8 7     1 4 7
*/
void anti_rotate(vector<vector<int> > &matrix) {
    for (auto vi : matrix) reverse(vi.begin(), vi.end());
    for (int i = 0; i < matrix.size(); ++i) {
        for (int j = i + 1; j < matrix[i].size(); ++j)
            swap(matrix[i][j], matrix[j][i]);
    }
}
```



## 1200. Minimum Absolute Difference

```python
class Solution:
    def minimumAbsDifference(self, arr: List[int]) -> List[List[int]]:
        arr = sorted(arr)
        diff = sys.maxsize
        ans = []
        for i in range(len(arr)-1):
            if arr[i+1] - arr[i] < diff:
                diff = arr[i+1] - arr[i] 
                ans = [[arr[i],arr[i+1]]]
            elif arr[i+1] - arr[i] == diff:
                ans.append([arr[i],arr[i+1]])
        return ans
```



## Count Numbers with Unique Digits

[leetcode](https://leetcode.com/problems/count-numbers-with-unique-digits/)

```python
class Solution:
    def countNumbersWithUniqueDigits(self, n: int) -> int:
        choices = [9, 9, 8, 7, 6, 5, 4, 3, 2, 1]
        ans, product = 1, 1
        
        for i in range(n if n <= 10 else 10):
            product *= choices[i]
            ans += product
            
        return ans
```

Backtracking solution

```python

```

DP solution

```python

```

## [Count Primes](https://leetcode.com/problems/count-primes/)

```python
class Solution:
    def countPrimes(self, n: int) -> int:
        if n < 3:
            return 0
        isPrime = [True]*n
        isPrime[0] = isPrime[1] = False
        
        # Loop's ending condition is i * i < n 
        for i in range(0, int(n**0.5)+1):
            if not isPrime[i]: 
                continue
                
            #The Sieve of Eratosthenes
            for j in range(i*i, n, i):
                isPrime[j] = False
        count = 0
        for x in isPrime:
            if x:
                count += 1
        return count
            
                
```

## Bits

### [Number of 1 Bits](https://leetcode.com/problems/number-of-1-bits/)

```python
class Solution:
    def hammingWeight(self, n: int) -> int:
        count = 0
        while n:
            n &= n-1
            count += 1
        return count
```

### [Counting Bits ](https://leetcode.com/problems/counting-bits/)

```python
class Solution:
    def countBits(self, num: int) -> List[int]:
        dp = [0]
        for i in range(1, num + 1):
            if i % 2 == 1:
                dp.append(dp[i - 1] + 1)
            else:
                dp.append(dp[i // 2])
        return dp
```

## mean, mod

```python
#mean
scores =  [91, 95, 97, 99, 92, 93, 96, 98]  

avg = sum(scores) / len(scores)

#方法二：
#导入函数库
import numpy as np

scores1 =  [91, 95, 97, 99, 92, 93, 96, 98]  
average = np.mean(scores1)  # 一行解决。

```

## 保留float类型小数点后3位

1 ’%.2f’ %f 方法\(推荐\)

```text
﻿﻿f = 1.23456

print('%.4f' % f)
print('%.3f' % f)
print('%.2f' % f)
```

2 format函数\(推荐\)

```text
print(format(1.23456, '.2f'))
print(format(1.23456, '.3f'))
print(format(1.23456, '.4f'))
123
```

```text
1.23
1.235
1.2346
```

`3.round()`

```text
>> x = 3.897654326
>> round(x, 3)
3.898
>> x = 3.000000
>> round(x, 3)
3.0
123456
```

`round`函数自动四舍五入；自动去掉多余的0

