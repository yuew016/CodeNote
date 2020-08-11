# 1.3 Binary Search

## 1.1 Defination

Given an sorted integer array - nums, and an integer - target.

Find the **any/first/last**\(maybe overtime\) position of target in nums Return **-1** if target does not exist.

### Time Complexity

T\(n\) = T\(n/2\) + O\(1\) = O\(logn\): 通过O\(1\)的时间，把规模为n的问题变为n/2: recursive

通过O\(n\)的时间，把规模为n的问题变为n/2?: O\(n\)

• O\(1\) 极少

• **O\(logn\)** 几乎都是二分法

• O\(√n\) 几乎是分解质因数

• **O\(n\)** 高频

**• O\(nlogn\)** 一般都可能要排序

**• O\(n2\)** 数组，枚举，动态规划

• **O\(n3\)** 数组，枚举，动态规划

• **O\(2n\)** 与组合有关的搜索

• **O\(n!\)** 与排列有关的搜索

**比O\(n\)更优的时间复杂度, 几乎只能是O\(logn\)的二分法**

### Recursion or Non-Recursion

• 面试中是否使用 Recursion 的几个判断条件

1. 面试官是否要求了不使用 Recursion \(如果你不确定，就向面试官询问\)
2. 不用 Recursion 是否会造成实现变得很复杂
3. Recursion 的深度是否会很深
4. 题目的考点是 Recursion vs Non-Recursion 还是就是考你是否会Recursion?

• 记住:不要自己下判断，要跟面试官讨论!

## 1.2 Templete

Last position: 相等时 start = mid, 只有该情况会出现死循环

First position: if ==: end = mid

```python
Binary search is a famous question in algorithm. For a given sorted array (ascending order) and a target number, find the last index of this number in O(log n) time complexity. If the target number does not exist in the array, return -1. 

Example If the array is [1, 2, 3, 3, 4, 5, 10], for given target 3, return 2.

#python
class Solution:
    """
    @param nums: An integer array sorted in ascending order
    @param target: An integer
    @return: An integer
    """
   def lastPosition(self, nums, target):
        # write your code here
        if nums == []:
            return -1
        start, end = 0, len(nums)-1
        while start +1 < end:
            mid = start+ (end - start)//2
            if nums[mid] == target:
                start = mid
            elif nums[mid] < target:
                start = mid
            else:
                end = mid
        if nums[end] == target:
            return end
        elif nums[start] == target:
            return start    
        return -1
```

## 1.3 Examples

二分位置 之 OOXX 一般会给你一个数组

让你找数组中第一个/最后一个满足某个条件的位置 OOOOOOO...O**O\*\***X\*\*X....XXXXXX

### First Bad Version

```python
class Solution:
    """
    @param n: An integers.
    @return: An integer which is the first bad version.
    """
    def findFirstBadVersion(self, n):
        start, end = 1, n
        while start + 1 < end:
            mid = (start + end) // 2
            if SVNRepo.isBadVersion(mid):
                end = mid
            else:
                start = mid
        if SVNRepo.isBadVersion(start):
            return start
        return end
```

### Search In a Big Sorted Array

```java
public class Solution {
    public int searchBigSortedArray(ArrayReader reader, int target) {
        int len = 1;
        while(reader.get(len - 1) != -1 && reader.get(len - 1) < target) 
          len *= 2; //倍增思想
        int i = len / 2, j = len - 1;
        while(i < j - 1) {
            int m = i + (j - i) / 2;
            if(reader.get(m) == -1 || reader.get(m) >= target) {
                j = m;
            } else {
                i = m;
            }
        }
        if(reader.get(i) == target) return i;
        if(reader.get(j) == target) return j;
        return -1;
    }
}
```

### Find Minimum in Rotated Sorted Array

```python
class Solution:
    """
    @param nums: a rotated sorted array
    @return: the minimum number in the array
    """
    def findMin(self, nums):
        # write your code here
        if not nums:
            return -1

        start, end=0, len(nums) -1
        while start+ 1<end:
            mid = start + (end-start) // 2
            if nums[mid] > nums[end]:
                start = mid
            else:
                end = mid
        return min(nums[start], nums[end])
```

### Search Insert Position

```python
#time O(logn) space O(1)
class Solution:
    def searchInsert(self, nums: List[int], target: int) -> int:
        s, e=0, len(nums)-1
        while s+1 < e:
            mid = s+(e-s)//2
            if target == nums[mid]:
                s = mid
            elif target < nums[mid]:
                e = mid
            else:
                s = mid
        if nums[s]>= target:
            return s
        elif nums[e]>= target:
            return e
        else:
            return len(nums)
```

### Search for a Range

```python
class Solution:
    """
    @param A: an integer sorted array
    @param target: an integer to be inserted
    @return: a list of length 2, [index1, index2]
    """
    def searchRange(self, A, target):
        # write your code here
        if A == []:
            return [-1, -1]
        s1,s2,e1,e2=0,0,len(A)-1,len(A)-1    
        while s1 + 1<e1:
            mid = s1 + (e1 - s1)//2
            if A[mid] == target:
                e1 = mid
            elif A[mid] < target:
                s1 = mid
            else:
                e1 = mid

        if A[s1] == target:
            index1 = s1
        elif A[e1] == target:
            index1 = e1
        else:
            return [-1, -1]

        while s2 + 1<e2:
            mid = s2 + (e2 - s2)//2
            if A[mid] == target:
                s2 = mid
            elif A[mid] < target:
                s2 = mid
            else:
                e2 = mid

        if A[e2] == target:
            index2 = e2
        elif A[s2] == target:
            index2 = s2

        return [index1, index2]
```

### Maximum Number in Mountain Sequence

在先增后减的序列中找最大值

```python
class Solution:
    """
    @param nums: a mountain sequence which increase firstly and then decrease
    @return: then mountain top
    """
    def mountainSequence(self, nums):
        # write your code here
        if not nums:
            return -1
        start, end = 0, len(nums)-1
        while start + 1 < end:
            mid = start + (end - start)//2
            if nums[mid] > nums[mid + 1]:
                end = mid
            else:
                start = mid
        return max(nums[start], nums[end])
```

### Find Peak Element

思路：如果中间元素大于其相邻后续元素，则中间元素左侧\(包含该中间元素）必包含一个局部最大值。如果中间元素小于其相邻后续元素，则中间元素右侧必包含一个局部最大值。

```python
class Solution:
    """
    @param A: An integers array.
    @return: return any of peek positions.
    """
    def findPeak(self, A):
        # write your code here
        if not A:
            return -1
        start, end =0, len(A) -1
        while start + 1<end:
            mid = start + (end - start)//2
            if A[mid] > A[mid+1]:
                end = mid
            else:
                start = mid
        if A[start] > A[end]:
            return start
        else:
            return end
```

### Search in Rotated Sorted Array

```python
class Solution:
    """
    @param A: an integer rotated sorted array
    @param target: an integer to be searched
    @return: an integer
    """
    def search(self, A, target):
        # write your code here
        if not A:
            return -1
        start, end = 0,len(A)-1
        while start + 1< end:
            mid = start + (end - start)//2
            if A[mid] >= A[start]:
                if A[start] <= target <= A[mid]:
                    end = mid
                else:
                    start = mid
            else:
                if A[mid] <= target <= A[end]:
                    start = mid
                else:
                    end = mid

        if A[start] == target:
            return start
        elif A[end] == target:
            return end
        return -1
```

### Search a 2D Matrix

```text

```

### Search a 2D Matrix II

```python
class Solution:
    def searchMatrix(self, matrix, target):
        """
        :type matrix: List[List[int]]
        :type target: int
        :rtype: bool
        """
        if not matrix:
            return False
        x,y = len(matrix)-1, 0
        while x>=0 and y< len(matrix[0]):
            if matrix[x][y] < target:
                y+=1
            elif matrix[x][y] > target:
                x-=1
            else:
                return True
        return False
```

### Rotate Array

三步翻转法: • \[4,5,1,2,3\] → \[5,4,1,2,3\] → \[5,4,3,2,1\] → \[1,2,3,4,5\]

Reverse:

time=O\(n\)

space=S\(1\)

```python
class Solution:
    """
    @param str: An array of char
    @param offset: An integer
    @return: nothing

    """ 

    def reverse_string(self,s, lo, hi):
        while lo <= hi:
            s[lo], s[hi] = s[hi], s[lo]
            lo += 1 
            hi -= 1     

    def rotateString(self, s, offset):

        if not s: return

        n = len(s)
        offset %= n
        if offset == 0: return 

        self.reverse_string(s, n - offset, n - 1)
        self.reverse_string(s, 0, n - offset - 1)
        self.reverse_string(s, 0, n-1)
```

### Recover Rotated Sorted Array

给定一个**旋转**排序数组，在原地恢复其排序。（升序）

算法课教程教的三步翻转。 注意！重要的事情说两遍： 找那个断点的地方不能用二分法！不要被楼上一些解答误导了！ 找那个断点的地方不能用二分法！不要被楼上一些解答误导了！ 套用二分法 find min in RSA的前提条件是：没有重复数！这题目遇到 1 1 1 1 1 1 1 1 1 0 1 1 1 1，用二分法会找错地方. 所以只能用打擂台！

```python
class Solution:
    """
    @param nums: An integer array
    @return: nothing
    """
    def recoverRotatedSortedArray(self, nums):
        # write your code here
        if not nums:
            return
        for mix in range(1, len(nums)):
            if nums[mix] < nums[mix - 1]:
                break
        else:
            return
        self.reverse(nums, 0, mix-1)
        self.reverse(nums, mix, len(nums)-1)
        self.reverse(nums, 0, len(nums)-1)

    def reverse(self, nums, start, end):
        while start < end:
            nums[start], nums[end] = nums[end], nums[start]
            start += 1 
            end -= 1
```

### Median of Two Sorted Array\(hard\)

### Smallest Rectangle Enclosing Black Pixels\(hard\)

### \744. Find Smallest Letter Greater Than Target

```python
class Solution:
    #15:15
    #binary search
    #time:O(logn) space:O(1)
    def nextGreatestLetter(self, letters: List[str], target: str) -> str:
        if target < letters[0]:
            return letters[0]
        elif target >= letters[-1]:
            return letters[0]
        l,r = 0, len(letters)-1
        while l + 1 < r:
            mid = (r + l)//2
            if target < letters[mid]:
                r = mid
            else:
                l = mid
        if target < letters[l]:
            return letters[l]
        else:
            return letters[r]
```

