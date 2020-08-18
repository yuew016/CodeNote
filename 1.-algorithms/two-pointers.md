# 1.1 Two Pointers

## •6.1 同向双指针

### \283. Move Zeroes

```python
class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        if not nums or not len(nums):
            return nums
        pos = 0
        for i in range(len(nums)):
            if nums[i] != 0:
                nums[pos] = nums[i]
                pos += 1
            i += 1
        while pos<len(nums):
            nums[pos] = 0
            pos += 1
```

### \1343. Number of Sub-arrays of Size K and Average Greater than or Equal to Threshold

```python
#scan window
#time: O[n]
#space: O(1)
class Solution:
    def numOfSubarrays(self, arr: List[int], k: int, threshold: int) -> int:
        if not arr or len(arr) < k:
            return 0
        count = 0
        sum = 0
        target = k*threshold
        left, win = 0, k
        for right, x in enumerate(arr):
            sum += x
            if right - left + 1 == win:
                if sum >= target:
                    count += 1
                sum -= arr[left]
                left += 1
        return count


def numOfSubarrays(self, a: List[int], k: int, threshold: int) -> int:
        lo, sum_of_win, cnt, target = -1, 0, 0, k * threshold
        for hi, v in enumerate(a):
            sum_of_win += v
            if hi - lo == k:
                if sum_of_win >= target:
                    cnt += 1
                lo += 1   
                sum_of_win -= a[lo]
        return cnt                



#prefix array
#time: O(n)
#space: O(n)
class Solution:
    def numOfSubarrays(self, arr: List[int], k: int, threshold: int) -> int:
        if not arr or len(arr) < k:
            return 0
        prefix = [0]
        target = k*threshold
        count = 0
        for i in arr:
            prefix.append(i + prefix[-1])
        for i in range(len(arr)-k+1):
            if prefix[i+k] - prefix[i] >= target:
                count += 1
        return count
```

### \1438. Longest Continuous Subarray With Absolute Diff Less Than or Equal to Limit

```python
#15:44
#two deque and two pointers
#time: O(n) space:O(n)
class Solution:
    def longestSubarray(self, nums: List[int], limit: int) -> int:
        maxd = collections.deque()
        mind = collections.deque()
        s = 0
        for x in nums:
            while maxd and maxd[-1] < x:
                maxd.pop()
            while mind and mind[-1] > x:
                mind.pop()
            maxd.append(x)
            mind.append(x)

            if maxd[0] - mind[0] > limit:
                if nums[s] ==maxd[0]: 
                    maxd.popleft()
                if nums[s] ==mind[0]: 
                    mind.popleft()
                s += 1
        return len(nums) - s
```

### \713. Subarray Product Less Than K

```python
#18:42
#two pointers //sliding window
#time: O(n) space:O(1)
class Solution:
    def numSubarrayProductLessThanK(self, nums: List[int], k: int) -> int:
        if k<=1:
            return 0
        l = 0
        product = 1
        ans = 0
        for r in range(len(nums)):
            product = product*nums[r]
            while product >= k and l<=r:
                product = product/nums[l]
                l += 1
            ans += r - l + 1
        return ans
```

\1089. Duplicate Zeros

```python
#two pointers
#time:O(n) space:O(1)
class Solution:
    def duplicateZeros(self, arr: List[int]) -> None:
        """
        Do not return anything, modify arr in-place instead.

        """
        n = len(arr)
        zeros = arr.count(0)
        for i in range(n-1, -1,-1):
            if i+zeros <n:
                arr[i+zeros] = arr[i]
            if arr[i] == 0:
                zeros -= 1
                if i+zeros<n:
                    arr[i+zeros] = 0


#one pass
class Solution:
    def duplicateZeros(self, arr: List[int]) -> None:
        """
        Do not return anything, modify arr in-place instead.
        """
        i = 0
        while i<len(arr):
            if arr[i] == 0:
                arr.insert(i+1, 0)
                del arr[len(arr)-1]
                i += 2
            else:
                i += 1
```

## **• 6.2 相向双指针**

### \125. Valid Palindrome

```python
class Solution:
    def isPalindrome(self, s: str) -> bool:
        s = ''.join([c for c in s if c.isalnum()]).lower()
        i, j = 0, len(s)-1
        while i < j:
            if s[i] != s[j]:
                return False
            i+= 1
            j -= 1
        return True
```

### \680. Valid Palindrome II

Given a non-empty string `s`, you may delete **at most** one character. Judge whether you can make it a palindrome.

```python
class Solution:
    def validPalindrome(self, s: str) -> bool:
        l, r = 0, len(s)-1
        count = 0

        def helper(l, r, count):
            if count > 1:
                return False
            while l<r:
                if s[l] == s[r]:
                    l += 1
                    r -= 1
                elif s[l+1] != s [r] and s[l] != s [r-1]:
                        return False
                else:
                    return helper(l+1, r, count + 1) or helper(l, r-1, count+1)
            return True

        return helper(l, r, count)
```

## **• 6.3 Two Sum**

​ • 几乎所有 Two Sum 变种

哈希表\(HashMap\) vs 两根指针\(Two Pointers\)

对于求 2 个变量如何组合的问题

可以循环其中一个变量，然后研究另外一个变量如何变化

### Two Sum III - Data structure design

```python
#hashMap
class TwoSum:
    """
    @param number: An integer
    @return: nothing
    """

    def __init__(self):
        # initialize your data structure here
        self.hashMap = {}

    def add(self, number):
        # write your code here
        if number in self.hashMap:
            self.hashMap[number] += 1 
        else:
            self.hashMap[number] = 1 

    """
    @param value: An integer
    @return: Find if there exists any pair of numbers which sum is equal to the value.
    """

    def find(self, value):
        # write your code here
        for key in self.hashMap:
            if value - key in self.hashMap and \
            (value - key != key or self.hashMap[key] > 1):
                return True 
        return False
```

### \167. Two Sum II - Input array is sorted

```python
class Solution:
    def twoSum(self, numbers: List[int], target: int) -> List[int]:
        i,j=0, len(numbers)-1
        while i<j:
            if numbers[i] + numbers[j] == target:
                return [i+1,j+1]
            elif numbers[i] + numbers[j] < target:
                i += 1
            else:
                j -= 1
```

### \653. Two Sum IV - Input is a BST

```text
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    #hash map+preorder 
    #time:O(n) space:O(n)
    def findTarget(self, root: TreeNode, k: int) -> bool:
        if not root:
            return False
        hash_set = set()
        return self.traverse(root, hash_set, k)

    def traverse(self,root, hash_set, k):
        if not root:
            return False
        elif k-root.val in hash_set :
            return True
        else:
            hash_set.add(root.val)
            return self.traverse(root.left, hash_set, k) \
        or self.traverse(root.right, hash_set, k)
```

### Two Sum - Unique pairs

```python
class Solution:
    """
    @param nums: an array of integer
    @param target: An integer
    @return: An integer
    """
    def twoSum6(self, nums, target):
        if not nums or len(nums) < 2:
            return 0

        nums.sort()

        count = 0
        left, right = 0, len(nums) - 1

        while left < right:
            if nums[left] + nums[right] == target:
                count, left, right = count + 1, left + 1, right - 1
                while left < right and nums[left] == nums[left - 1]:
                    left += 1
                while left < right and nums[right] == nums[right + 1]:
                    right -= 1
            elif nums[left] + nums[right] > target:
                right -= 1
            else:
                left += 1

        return count


class Solution:
    """
    @param nums: an array of integer
    @param target: An integer
    @return: An integer
    """
    def twoSum6(self, nums, target):
        if not nums or len(nums) < 2:
            return 0

        nums.sort()

        count = 0
        left, right = 0, len(nums) - 1
        last_pair = (None, None)

        while left < right:
            if nums[left] + nums[right] == target:
                if (nums[left], nums[right]) != last_pair:
                    count += 1
                last_pair = (nums[left], nums[right])
                left, right = left + 1, right - 1
            elif nums[left] + nums[right] > target:
                right -= 1
            else:
                left += 1

        return count
```

### \15. 3Sum

```python
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        if not nums or len(nums) < 3:
            return []
        results = []
        nums.sort()
        for i in range(len(nums)-2):
            if i > 0 and nums[i] == nums[i-1]:
                continue
            self.find_two_sum(nums[i+1:], results, -nums[i])
        return results

    def find_two_sum(self, nums, results, target):
        l,r = 0, len(nums)-1
        while l < r:
            if nums[l] + nums[r] == target:
                results.append([-target, nums[l], nums[r]])
                l += 1
                r -= 1
                while l<r and nums[l] == nums[l-1]:
                    l += 1
                while l<r and nums[r] == nums[r+1]:
                    r -= 1
            elif nums[l] + nums[r] > target:
                r -= 1
            else:
                l += 1
```

### \16. 3Sum Closest

```python
class Solution:
    #14:47
    #排序后。 固定一个点，利用双指针的方式，扫描，记录答案即可。
    def threeSumClosest(self, nums: List[int], target: int) -> int:
        nums.sort()
        ans = None
        for i in range(len(nums)-2):
            l, r = i+1, len(nums)-1
            while l<r:
                sum = nums[i] + nums[l] + nums[r]
                if ans is None or abs(sum-target) < abs(ans - target):
                    ans = sum
                if sum < target:
                    l += 1
                elif sum > target:
                    r -= 1
                else:
                    return ans
        return ans
```

### \923. 3Sum With Multiplicity

```python
#15:43
#Approach 1: Three Pointer TLE
#Time Complexity: O(N^2)O(N^2).
#Space Complexity: O(1)O(1).
class Solution:
    def threeSumMulti(self, A: List[int], target: int) -> int:
        count = 0
        MOD = 10**9 + 7
        A.sort()

        for i in range(len(A) - 2):
            l, r = i+1, len(A)-1
            while l < r:
                sum = A[i] + A[l] + A[r]
                if sum < target:
                    l += 1
                elif sum > target:
                    r -= 1
                else:
                    if A[l] == A[r]:
                        count += (r-l+1)*(r-l)//2
                        count %= MOD
                        break
                    else:
                        a, b =1, 1
                        while l + 1 < r and A[l+1] == A[l]:
                            a += 1
                            l += 1
                        while l - 1 < r and A[r-1] == A[r]:
                            b += 1
                            r -= 1
                        count += a*b
                        count %= MOD
                        l += 1
                        r -= 1           
        return count



#Approach 2: Three Pointer + hash set
#Time Complexity: O(N^2)O(N^2).
#Space Complexity: O(N).
class Solution:
    def threeSumMulti(self, A: List[int], target: int) -> int:
        MOD = 10**9 + 7
        count = collections.Counter(A)
        keys = sorted(count)

        ans = 0

        # Now, let's do a 3sum on "keys", for i <= j <= k.
        # We will use count to add the correct contribution to ans.
        for i, x in enumerate(keys):
            T = target - x
            j, k = i, len(keys) - 1
            while j <= k:
                y, z = keys[j], keys[k]
                if y + z < T:
                    j += 1
                elif y + z > T:
                    k -= 1
                else: # x+y+z == T, now calculate the size of the contribution
                    if i < j < k:
                        ans += count[x] * count[y] * count[z]
                    elif i == j < k:
                        ans += count[x] * (count[x] - 1) // 2 * count[z]
                    elif i < j == k:
                        ans += count[x] * count[y] * (count[y] - 1) // 2
                    else:  # i == j == k
                        ans += count[x] * (count[x] - 1) * (count[x] - 2) // 6

                    j += 1
                    k -= 1

        return ans % MOD
```

### \18. 4Sum

```python
#two sum
#time:O(n3) space:O(1)
class Solution:
    def fourSum(self, nums: List[int], target: int) -> List[List[int]]:
        def findNsum(l, r, target, N, result, results):
            if r-l+1 < N or N < 2 or target < nums[l]*N or target > nums[r]*N:  
                # early termination
                return
            if N == 2: # two pointers solve sorted 2-sum problem
                while l < r:
                    s = nums[l] + nums[r]
                    if s == target:
                        results.append(result + [nums[l], nums[r]])
                        l += 1
                        while l < r and nums[l] == nums[l-1]:
                            l += 1
                    elif s < target:
                        l += 1
                    else:
                        r -= 1
            else: # recursively reduce N
                for i in range(l, r+1):
                    if i == l or (i > l and nums[i-1] != nums[i]):
                        findNsum(i+1, r, target-nums[i], N-1, result+[nums[i]], results)


        nums.sort()
        results = []
        findNsum(0, len(nums)-1, target, 4, [], results)
        return results

#k sum    
#two sum
#time:O(n3) space:O(1)
class Solution:
    def fourSum(self, nums: List[int], target: int) -> List[List[int]]:
        def findNsum(l, r, target, N, result, results):
            if r-l+1 < N or N < 2 or target < nums[l]*N or target > nums[r]*N:  
                # early termination
                return
            if N == 2: # two pointers solve sorted 2-sum problem
                while l < r:
                    s = nums[l] + nums[r]
                    if s == target:
                        results.append(result + [nums[l], nums[r]])
                        l += 1
                        while l < r and nums[l] == nums[l-1]:
                            l += 1
                    elif s < target:
                        l += 1
                    else:
                        r -= 1
            else: # recursively reduce N
                for i in range(l, r+1):
                    if i == l or (i > l and nums[i-1] != nums[i]):
                        findNsum(i+1, r, target-nums[i], N-1, result+[nums[i]], results)


        nums.sort()
        results = []
        findNsum(0, len(nums)-1, target, 4, [], results)
        return results
```

### \454. 4Sum II

```python
#hash set
#time:O(n2) sapce:O(n2)
class Solution:
    def fourSumCount(self, A: List[int], B: List[int], C: List[int], D: List[int]) -> int:
        ans = 0
        hashSum = {}
        for a in A:
            for b in B:
                if a+b in hashSum:
                    hashSum[a+b] += 1
                else:
                    hashSum[a+b] = 1
        for c in C:
            for d in D:
                target = -(c+d)
                if target in hashSum:
                    ans += hashSum[target]
        return ans
```

### \611. Valid Triangle Number

```python
#3sum
#three pointers scan
#time O(n^2)  s
#space: O(1)
class Solution:
    def triangleNumber(self, nums: List[int]) -> int:
        nums.sort()

        ans = 0
        for i in range(len(nums)):
            left, right = 0, i - 1
            while left < right:
                if nums[left] + nums[right] > nums[i]:
                    ans += right - left
                    right -= 1
                else:
                    left += 1
        return ans
```

Two Sum 计数问题

### Two Sum &lt;= target

O\(1\) 额外空间以及 O\(nlogn\) 时间复杂度

```python
class Solution:
    # @param nums {int[]} an array of integer
    # @param target {int} an integer
    # @return {int} an integer
    def twoSum5(self, nums, target):
        # Write your code here
        l, r = 0, len(nums)-1
        cnt = 0
        nums.sort()
        while l < r:
            value = nums[l] + nums[r]
            if value > target:
                r -= 1
            else:
                cnt += r - l
                l += 1
        return cnt
```

### two sum &gt;= target

```python
class Solution:
    # @param nums {int[]} an array of integer
    # @param target {int} an integer
    # @return {int} an integer
    def twoSum5(self, nums, target):
        # Write your code here
        l, r = 0, len(nums)-1
        cnt = 0
        nums.sort()
        while l < r:
            value = nums[l] + nums[r]
            if value > target:
                cnt += r - l
                r -= 1
            else:
                l += 1
        return cnt
```

## 6.4 **Partition**

​ • Quick Select • 分成两个部分 • 分成三个部分 • 一些你没听过的\(但是面试会考的\)排序算法

### \561. Array Partition I

```python
''''
Consider the smallest element x. It should be paired with the next smallest element, because min(x, anything) = x, and having bigger elements only helps you have a larger score. Thus, we should pair adjacent elements together in the sorted array.
''''
class Solution:
    def arrayPairSum(self, nums: List[int]) -> int:
        mysum=0
        nums.sort()
        for i in range(0, len(nums), 2):
            mysum += nums[i]
        return mysum
```

### \905. Sort Array By Parity

```python
class Solution:
    def sortArrayByParity(self, A: List[int]) -> List[int]:
        l, r = 0, len(A)-1
        while l <= r:
            while l <= r and A[l]%2 == 0:
                l += 1
            while l <= r and A[r]%2 == 1:
                r -= 1
            if l <= r:
                A[l], A[r] = A[r], A[l] 
                l += 1
                r -= 1
        return A
```

### Interleaving Positive and Negative Numbers

给出一个含有正整数和负整数的数组，重新排列成一个正负数交错的数组。

不需要保持正整数或者负整数原来的顺序。

```python
class Solution:
    """
    @param: A: An integer array.
    @return: nothing
    """
    def rerange(self, A):
        # write your code here
        if not A or len(A) <= 2:
            return A

        pos = len([a for a in A if a > 0])
        neg = len(A) - pos

        self.partition(A, pos > neg)
        self.interleave(A, pos == neg)

        #the longer one to head
    def partition(self, A, start_positive):
        flag = 1 if start_positive else -1
        l, r = 0, len(A)-1
        while l <= r:
            while l <= r and A[l]*flag > 0:
                l += 1 
            while l <= r and A[r]*flag < 0:
                r -= 1 
            if l <= r:
                A[l], A[r] = A[r], A[l]
                l += 1 
                r -= 1 

    def interleave(self, A, same_length):
        left, right = 1, len(A) - 1 
        if same_length:
            right = len(A) - 2 
        while left < right:
            A[left], A[right] = A[right], A[left]
            left += 2
            right -= 2
```

### Sort Letters by Case

Given a string which contains only letters. Sort it by lower case first and upper case second.

```python
#partition 2
class Solution:
    """
    @param: chars: The letter array you should sort by Case
    @return: nothing
    """
    def sortLetters(self, chars):
        # write your code here
        if not chars or len(chars)<=1:
            return chars

        left, right = 0, len(chars) - 1 
        while left <= right:
            while left <= right and 'a' <= chars[left] <= 'z':
                left += 1 
            while left <= right and 'A' <= chars[right] <= 'Z':
                right -= 1 
            if left <= right:
                chars[left], chars[right] = chars[right], chars[left]
                left += 1 
                right -= 1 
        return chars
```

### Sort Colors

分成三个部分:左中右

V1：两个循环，先分成左，中右；再分成中，右

V2: 统计各类别的个数（counting sort\)【必须：可数】

V3:三分法 one-pass algorithm using only constant space

```python
#one-pass algorithm using only constant space
#partition 3p 
#16:39
class Solution:
    def sortColors(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        if not nums or not len(nums):
            return 
        l, m, r = 0, 0, len(nums)-1 
        while m <= r :
            if nums[m] == 0:
                nums[l], nums[m] = nums[m], nums[l]
                l += 1
                m += 1
            elif nums[m] == 2:
                nums[r], nums[m] = nums[m], nums[r]
                r -= 1
            else:
                m += 1
```

### Rainbow Sort

```python
#time:O(nlogk)
class Solution:
    """
    @param colors: A list of integer
    @param k: An integer
    @return: nothing
    """
    def sortColors2(self, colors, k):
        # write your code here
        if not colors or k <=1:
            return
        self.rainbowSort(colors, 1, k, 0, len(colors)-1)

    def rainbowSort(self, colors, color_from, color_to, index_from, index_to):
        if color_to == color_from or index_from == index_to:
            return
        mid = (color_to + color_from) // 2 
        l, r = index_from, index_to
        while l <= r:
            while l <= r and colors[l] <= mid :
                l += 1 
            while l <= r and colors[r] > mid :
                r -= 1 
            if l<=r:
                colors[l], colors[r] = colors[r], colors[l]
                l += 1 
                r -= 1 
        self.rainbowSort(colors, color_from, mid, index_from, r)
        self.rainbowSort(colors, mid+1, color_to, l, index_to)
```

### Pancake Sort

Let given array be arr\[\] and size of array be n. 1\) Start from current size equal to n and reduce current size by one while it’s greater than 1. Let the current size be curr\_size. Do following for every curr\_size ……a\) Find index of the maximum element in arr\[0..curr\_szie-1\]. Let the index be ‘mi’ ……b\) Call flip\(arr, mi\) ……c\) Call flip\(arr, curr\_size-1\)

```python
# Python3 program to 
# sort array using 
# pancake sort 

# Reverses arr[0..i] */ 
def flip(arr, i): 
    start = 0
    while start < i: 
        temp = arr[start] 
        arr[start] = arr[i] 
        arr[i] = temp 
        start += 1
        i -= 1

# Returns index of the maximum 
# element in arr[0..n-1] */ 
def findMax(arr, n): 
    mi = 0
    for i in range(0,n): 
        if arr[i] > arr[mi]: 
            mi = i 
    return mi 

# The main function that 
# sorts given array 
# using flip operations 
def pancakeSort(arr, n): 

    # Start from the complete 
    # array and one by one 
    # reduce current size 
    # by one 
    curr_size = n 
    while curr_size > 1: 
        # Find index of the maximum 
        # element in 
        # arr[0..curr_size-1] 
        mi = findMax(arr, curr_size) 

        # Move the maximum element 
        # to end of current array 
        # if it's not already at 
        # the end 
        if mi != curr_size-1: 
            # To move at the end, 
            # first move maximum 
            # number to beginning 
            flip(arr, mi) 

            # Now move the maximum 
            # number to end by 
            # reversing current array 
            flip(arr, curr_size-1) 
        curr_size -= 1

# A utility function to 
# print an array of size n 
def printArray(arr, n): 
    for i in range(0,n): 
        print ("%d"%( arr[i]),end=" ") 

# Driver program 
arr = [23, 10, 20, 11, 12, 6, 7] 
n = len(arr) 
pancakeSort(arr, n); 
print ("Sorted Array ") 
printArray(arr,n) 

# This code is contributed by shreyanshi_arun.
```

O\(n\) flip operations are performed in above code. The overall time complexity is O\(n^2\).

### Sleep Sort

Time: _O_\(_n_\) space: _O_\(_n_\)

```python
from time import sleep
from threading import Timer

def sleepsort(values):
    sleepsort.result = []
    def add1(x):
        sleepsort.result.append(x)
    mx = values[0]
    for v in values:
        if mx < v: mx = v
        Timer(v, add1, [v]).start()
    sleep(mx+1)
    return sleepsort.result

if __name__ == '__main__':
    x = [3,2,4,7,3,6,9,1]
    if sleepsort(x) == sorted(x):
        print('sleep sort worked for:',x)
    else:
        print('sleep sort FAILED for:',x)
```

### Spaghetti Sort

Time: _O_\(_n_\) space: _O_\(_n_\)

### Bogo Sort

Random sort

## 6.5 Quick Select

### \215. Kth Largest Element in an Array

```python
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        if not nums or not len(nums):
            return None
        return self.quickSort(nums, 0, len(nums)-1, k)

    def quickSort(self, nums, start, end, k):
        if start == end:
            return nums[start]
        s, e = start, end
        pivot = nums[(s+e)//2] 
        while s <= e:
            while s<=e and nums[s] > pivot:
                s += 1
            while s<=e and nums[e] < pivot:
                e -= 1 
            if s<=e:
                nums[s], nums[e] = nums[e], nums[s]
                s+=1
                e-=1

        if start + k - 1 <= e:
            return self.quickSort(nums, start, e, k)
        elif start + k - 1 >= s:
            return self.quickSort(nums, s, end, k-s+start)
        else:
            return nums[e+1]
```

### Median

Given a unsorted array with integers, find the median of it.

A median is the middle number of the array after it is sorted.

If there are even numbers in the array, return the `N/2`-th number after sorted.

```python
#quick sort 
#time:O(n) space:O(1)
class Solution:
    """
    @param nums: A list of integers
    @return: An integer denotes the middle number of the array
    """

    def median(self, nums):
        # write your code here
        if not nums or not len(nums):
            return None 

        return self.quickFindKth(nums, 0, len(nums)-1, (len(nums)+1)//2)

    def quickFindKth(self, nums, start, end, k):
        if start == end:
            return nums[start]
        l, r = start, end 
        pivot = nums[(l+r)//2]
        while l<=r:
            while l<=r and nums[l] < pivot:
                l += 1 
            while l<=r and nums[r] > pivot:
                r -= 1 
            if l<=r:
                nums[l], nums[r] = nums[r], nums[l]
                l += 1 
                r -= 1 
        if start + k - 1 <= r:
            return self.quickFindKth(nums, start, r, k)
        elif start + k - 1 >= l:
            return self.quickFindKth(nums, l, end, k-(l-start))
        else:
            return nums[r+1]
```

