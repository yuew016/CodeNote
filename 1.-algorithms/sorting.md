# 1.4 Sorting

## Sorting排序算法

### complexity

稳定\*\*：如果a原本在b前面，而a=b，排序之后a仍然在b的前面。

### coding

### \#Bubble Sort

```python
def sortIntegers(self, A):
        # write your code here
        for i in range(len(A)-1):
            for j in range(len(A) - 1 - i):
                if A[j] > A[j+1]:
                A[j], A[j + 1] = A[j + 1], A[j]
```

### \#Selection Sort

```python
def sortIntegers(self, A):
        # write your code here
        for i in range(len(A)-1):
            minIndex = i 
            for j in range(i+1, len(A)):
                if A[j] < A[minIndex]:
                    minIndex = j 
            A[i], A[minIndex] = A[minIndex], A[i] 
        return A
```

### \#Insertion Sort

```python
def sortIntegers(self, A):
        # write your code here
        for i in range(len(A)-1):
            curnum = A[i+1] #num for insert
            preIn = i #position for insert
            while preIn >= 0 and A[preIn] > curnum:
                A[preIn + 1 ]= A[preIn]
                preIn -= 1 
            A[preIn + 1] = curnum
        return A
```

### \#Shell Sort

```python
def sortIntegers(self, A):
        # write your code here
        gap = 1 
        while gap < len(A) //3:
            gap = gap*3+1 #dynamic gap
        while gap > 0:
            for i in range(gap, len(A)):
                curNum, preIndex = A[i], i - gap  # curNum 保存当前待插入的数
                while preIndex >= 0 and curNum < A[preIndex]:
                    A[preIndex + gap] = A[preIndex] # 将比 curNum 大的元素向后移动
                    preIndex -= gap
                A[preIndex + gap] = curNum  # 待插入的数的正确位置
            gap //= 3  # 下一个动态间隔
        return A
```

### \#Merge Sort

```python
def sortIntegers(self,A):
        if len(A) <= 1: return A
        middle = len(A) // 2 
        A1 = A[:middle]
        A2 = A[middle:]
        self.sortIntegers(A1)
        self.sortIntegers(A2)
        k = 0
        while len(A1) and len(A2):
            if A1[0] < A2[0]: A[k]=A1.pop(0)
            else: A[k]=A2.pop(0)
            k = k + 1

        while len(A1):
            A[k] = A1.pop(0)
            k = k + 1
        while len(A2):
            A[k] = A2.pop(0)
            k = k + 1   

#Merge sort v2
class Solution:
    """
    @param A: an integer array
    @return: nothing
    """
    def sortIntegers2(self,A):
        if A is None or len(A)==0:
            return
        tmp = [0] * len(A)
        self.mergeSort(A,0,len(A)-1, tmp)

    def mergeSort(self, A, start, end, tmp):
        if start >= end:
            return
        mid = (end + start)//2 
        self.mergeSort(A, start, mid, tmp)
        self.mergeSort(A, mid + 1, end, tmp)
        self.merge(A, start, end, tmp)

    def merge(self, A, start, end, tmp):
        mid = (end + start)//2
        leftIndex = start   
        rightIndex = mid + 1 
        Index = leftIndex 

        while leftIndex<= mid and rightIndex <= end :
            if A[leftIndex] < A[rightIndex]:
                tmp[Index] = A[leftIndex]
                leftIndex += 1 
            else:     
                tmp[Index] = A[rightIndex]
                rightIndex += 1 
            Index += 1     

        while leftIndex <= mid :
            tmp[Index] = A[leftIndex]
            Index += 1 
            leftIndex += 1 

        while rightIndex <= end :
            tmp[Index] = A[rightIndex]    
            Index += 1 
            rightIndex += 1 

        for i in range(start, end+1):
            A[i] = tmp[i]
```

### \#quick sort

```python
def sortIntegers2(self, A):
        # write your code here
        if not A or len(A) == 0:
            return
        self.quickSort(A, 0, len(A)-1)

    def quickSort(self, A, start, end):
        if start >= end:
            return

        left, right = start, end 
        # key point 1: pivot is the value, not the index!!!
        pivot = A[(start + end)//2] 
        # key point 2: every time, it should be left <= right not left < right 
        # otherwise, it'll exit at l=r,and each half include the same number,it'll stack overflow
        while left <= right:
                #key point 3:A[left] < pivot not <=, for more mean (1111112)
            while left <= right and A[left] < pivot: 
                left +=1 
            while left <= right and A[right] > pivot:
                right -=1 

            if left <= right:
                A[left], A[right] = A[right], A[left]
                left += 1 
                right -= 1 
        self.quickSort(A, start, right)
        self.quickSort(A, left, end)
```

### \#Heap Sort

```python
def sortIntegers2(self, A):
        # write your code here
        if not A or len(A) == 0:
            return
        size = len(A)
        self.buildHeap(A)
        for i in range(len(A))[::-1]:
            A[0], A[i] = A[i], A[0]
            self.adjustHeap(A,0,i)


    def buildHeap(self, A):
        size = len(A)
        if size <= 1:
            return
        for i in range(size//2)[::-1]:
            self.adjustHeap(A,i,size)

    def adjustHeap(self, A, start, size):
        if start >= size:
            return
        leftchild = start*2 + 1 
        rightchild = start*2 + 2 
        largest = start
        if leftchild < size and A[leftchild] > A[largest]:
            largest = leftchild
        if rightchild < size and A[rightchild] > A[largest]:
            largest = rightchild
        if largest != start:
            A[largest], A[start] = A[start], A[largest]
            self.adjustHeap(A, largest, size)
```

### \#Counting Sort

只能排自然数

```python
def countingSort(nums):
    bucket = [0] * (max(nums) + 1) # 桶的个数
    for num in nums:  # 将元素值作为键值存储在桶中，记录其出现的次数
        bucket[num] += 1
    i = 0  # nums 的索引
    for j in range(len(bucket)):
        while bucket[j] > 0:
            nums[i] = j
            bucket[j] -= 1
            i += 1
    return nums
```

### \#Bucket Sort

```python
def bucketSort(nums, defaultBucketSize = 5):
    maxVal, minVal = max(nums), min(nums)
    bucketSize = defaultBucketSize  # 如果没有指定桶的大小，则默认为5
    bucketCount = (maxVal - minVal) // bucketSize + 1  # 数据分为 bucketCount 组
    buckets = []  # 二维桶
    for i in range(bucketCount):
        buckets.append([])
    # 利用函数映射将各个数据放入对应的桶中
    for num in nums:
        buckets[(num - minVal) // bucketSize].append(num)
    nums.clear()  # 清空 nums
    # 对每一个二维桶中的元素进行排序
    for bucket in buckets:
        insertionSort(bucket)  # 假设使用插入排序
        nums.extend(bucket)    # 将排序好的桶依次放入到 nums 中
    return nums
```

### \#Radix Sort

基数排序须知：

基数排序是桶排序的一种推广，它所考虑的待排记录包含不止一个关键字。例如对一副牌的整理，可将每张牌看作一个记录，包含两个关键字：花色、面值。一般我们可以将一个有序列是先按花色划分为四大块，每一块中又再按面值大小排序。这时“花色”就是一张牌的“最主位关键字”，而“面值”是“最次位关键字”。

基数排序有两种方法：

1. MSD （主位优先法）：从高位开始进行排序
2. LSD （次位优先法）：从低位开始进行排序

作者：牛奶芝麻 链接：[https://www.jianshu.com/p/bbbab7fa77a2](https://www.jianshu.com/p/bbbab7fa77a2) 来源：简书 著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

```python
  # LSD Radix Sort
def radixSort(nums):
    mod = 10
    div = 1
    mostBit = len(str(max(nums)))  # 最大数的位数决定了外循环多少次
    buckets = [[] for row in range(mod)] # 构造 mod 个空桶
    while mostBit:
        for num in nums:  # 将数据放入对应的桶中
            buckets[num // div % mod].append(num)
        i = 0  # nums 的索引
        for bucket in buckets:  # 将数据收集起来
            while bucket:
                nums[i] = bucket.pop(0) # 依次取出
                i += 1
        div *= 10
        mostBit -= 1
    return nums
```

##  [Diagonal Traverse II](https://leetcode.com/problems/diagonal-traverse-ii/)

```python
#bfs
#time:O(n), space:(1)
import collections
class Solution:
    def findDiagonalOrder(self, nums: List[List[int]]) -> List[int]:
        res = []
        que = deque([(0,0)])
        while que:
            i,j = que.popleft()
            res.append(nums[i][j])
            if j == 0 and i+1<len(nums):
                que.append((i+1,j))
            if j+1<len(nums[i]):
                que.append((i,j+1))
        return res



#hash + deque
#time:O(n), space:(N)
import collections
class Solution:
    def findDiagonalOrder(self, nums: List[List[int]]) -> List[int]:
        dic = defaultdict(deque)
        for i,row in enumerate(nums):
            for j, ele in enumerate(row):
                dic[i+j].appendleft(ele)
        res = []
        for li in dic.values():
            res += li
        return res
```

