# 3. Math

## Matrix

小陷阱：初始化矩阵不能写成：

```python
before = [[0]*cols]*rows
```

而必须是：

```python
before = [[0]*cols for i in range(rows)]
or
before = [[0 for _ in range(cols)] for i in range(rows)]
```

### **A common method to rotate the image**

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

### [Reaching Points](https://leetcode.com/problems/reaching-points/)

```python
class Solution:
    def reachingPoints(self, sx: int, sy: int, tx: int, ty: int) -> bool:
        while sx < tx and sy < ty:
            tx, ty = tx % ty, ty % tx
        if sx == tx and sy <= ty and (ty - sy) % sx == 0:
            return True
        if sy == ty and sx <= tx and (tx - sx) % sy == 0:
            return True
        return False
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

## Bits位运算

**1、 判断整型的奇偶性**

使用位运算操作如下

```text
if((x & 1) == 0) {
    // 偶数
} else {
    // 奇数
}
```

这个例子相信大家都见过，只需判断整型的第一位是否是 1 即可，如果是说明是奇数，否则是偶数

**2、 判断第 n 位是否设置为 1**

```text
if (x & (1<<n)) {
    // 第 n 位设置为 1
}
else {
    // 第 n 位设置为 1
}
```

在上例中我们判断第一位是否为 1，所以如果要判断第 n 位是否 1，只要把 1 左移 n 位再作与运算不就完了。

**3、 将第 n 位设置为 1**

```text
y = x | (1<<n)
```

思路同第二步，先把 1 移到第 n 位再作或运算，这样第 n 位就肯定为 1。

**4、 将第 n 位设置为 0**

```text
y = x & ~(1<<n)
```

先将 1 左移到 第 n 位，再对其取反，此时第 n 位为 0，其他位都为 1，这样与 x 作与运算后，x 的第 n 位肯定为 0。

**5. 将第 n 位的值取反**

```text
y = x ^ (1<<n)
```

我们知道异或操作是两个数的每一位相同，结果为 0，否则是 1，所以现在把 1 左移到第 n 位，则如果 x 的第 n 位为 1，两数相同结果 0，如果 x 的第 n 位为 0，两数不相同，则为 1。来看个简单的例子

```text
    01110101
^   00100000
    --------
    01010101
```

如图示，第五位刚好取反

**6、 将最右边的 1 设为 0**

```text
y = x & (x-1)
```

如果说上面的 5 点技巧有点无聊，那第 6 条技巧确实很有意思，也是在 leetcode 经常出现的考点，下文中大部分习题都会用到这个知识点，务必要谨记！掌握这个很重要，有啥用呢，比如我要统计 1 的位数有几个，只要写个如下循环即可，不断地将 x 最右边的 1 置为 0，最后当值为 0 时统计就结束了。

```text
count = 0
while(x) {
  x = x & (x - 1);
  count++;
}
```

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

## power of two

```python
class Solution:
    def isPowerOfTwo(self, n: int) -> bool:
        if n<= 0:
            return False
        return not (n&(n-1))
```

## [Smallest power of 2 greater than or equal to n](https://www.geeksforgeeks.org/smallest-power-of-2-greater-than-or-equal-to-n/)

```python
def nextPowerOf2(n): 
  
    p = 1
    if (n and not(n & (n - 1))): 
        return n 
  
    while (p < n) : 
        p <<= 1
          
    return p; 
  
  
# Driver Code 
n = 5
print(nextPowerOf2(n)); 
```

## 415 Add Strings\(模拟加法运算）

```python
#18:26
#straight math method
class Solution:
    def addStrings(self, num1: str, num2: str) -> str:
        ans = []
        num1,num2 = list(num1),list(num2)
        carry = 0
        while len(num1)>0 or len(num2)>0:
            d1 = ord(num1.pop()) - ord('0') if len(num1)>0 else 0
            d2 = ord(num2.pop()) - ord('0') if len(num2)>0 else 0
            temp = d1+d2+carry
            carry = (temp)//10
            ans.append(temp%10)
        if carry:
            ans.append(carry)
        return "".join([str(i) for i in ans[::-1]])

            
            
```

