# 2.3 Hash & Heap

## Hash

### • 原理:

**Key, value, storage**

Hash Table / multithreading, thread safe

​ 多线程并发访问同一个hash表时安全

Hash Map /

Hash Set

支持操作:O\(1\) Insert / O\(1\) Find / O\(1\) Delete

O\(size of key\)

实现方法：Hash Function

使命:对于任意的key 得到一个固定且无规律的介于_0~capacity-1_的整数

一些著名的Hash算法 • MD5• SHA-1 • SHA-2

Magic Number - 31经验值

**Collision**

Open Hashing vs Closed Hashing

再好的 hash 函数也会存在冲突\(Collision\)

Closed:

Linear Probing: f\(i\) = i

Quadratic Probing: f\(i\) = i \* i

Double Hashing: f\(i\) = i \* hash2\(elem\)

Open :duck:

linked list

当hash不够大时怎么办?

饱和度 = 实际存储元素个数 / 总共开辟的空间大小 size / capacity 一般来说，超过 1/10\(经验值\) 的时候，说明需要进行 rehash

### Rehashing: lintcode

```python
"""
Definition of ListNode
class ListNode(object):

    def __init__(self, val, next=None):
        self.val = val
        self.next = next
"""
class Solution:
    """
    @param hashTable: A list of The first node of linked list
    @return: A list of The first node of linked list which have twice size
    """
    def rehashing(self, hashTable):
        # write your code here
        if len(hashTable) <= 0:
            return hashTable

        HASHSIZE = 2*len(hashTable)
        anshashTable = [None for i in range(HASHSIZE)]
        for item in hashTable:
            p = item
            while p != None:
                self.addnode(anshashTable, p.val)
                p = p.next
        return anshashTable

    def addnode(self, hashTable, key):
        code = key%len(hashTable)
        if hashTable[code] is None:
            hashTable[code] = ListNode(key)
        else:
            self.addlistnode(hashTable[code], key)

    def addlistnode(self, node, number):
        if node.next != None:
            self.addlistnode(node.next, number)
        else:
            node.next = ListNode(number)
```

### • 应用

### ⭐️LRU Cache

```python
class LinkNode:
    def __init__(self, key = None, value=None, next = None):
        self.key = key
        self.value = value 
        self.next = next 

class LRUCache:
    """
    @param: capacity: An integer
    """
    def __init__(self, capacity):
        # do intialization if necessary
        self.key_to_pre = {}
        self.head = LinkNode(0)
        self.tail = self.head
        self.capacity = capacity

    """
    @param: key: An integer
    @return: An integer
    """
    def moveback(self, pre):
        node = pre.next 
        if node == self.tail:
            return
        pre.next = node.next
        self.key_to_pre[node.next.key] = pre 
        #move to tail
        self.tail.next = node 
        self.key_to_pre[node.key] = self.tail 
        node.next = None 
        self.tail = node

    def get(self, key):
        # write your code here
        if key in self.key_to_pre:
            pre = self.key_to_pre[key]
            cur = pre.next
            self.moveback(pre)
            return cur.value
        return -1

    """
    @param: key: An integer
    @param: value: An integer
    @return: nothing
    """
    def pop_head(self):
        first = self.head.next
        del self.key_to_pre[first.key]
        if first.next:
            self.head.next = first.next
            self.key_to_pre[first.next.key] = self.head 
        else:
            self.head.next = None 



    def set(self, key, value):
        # write your code here
        #if exit key 
        if key in self.key_to_pre:
            self.moveback(self.key_to_pre[key])
            self.tail.value = value
            return

        #check full 
        if len(self.key_to_pre) == self.capacity:
            self.pop_head()
        self.tail.next = LinkNode(key, value)
        self.key_to_pre[key] = self.tail 
        self.tail = self.tail.next
```

### \128. Longest Consecutive Sequence

```python
class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        ans = 0
        hash_set = set(nums)
        for num in hash_set:
            if num-1 not in hash_set:
                cur = num
                cur_length = 1

                while cur + 1 in hash_set:
                    cur_length += 1
                    cur = cur+1

                ans = max(ans, cur_length)

        return ans
```

### \242. Valid Anagram

```python
#version1: sort
class Solution:
    def isAnagram(self, s: str, t: str) -> bool:
        if len(s) != len(t):
            return False
        s = list(s)
        t = list(t)
        s.sort()
        t.sort()
        return s == t

#version2: hash/dict
class Solution:
    def isAnagram(self, s: str, t: str) -> bool:
        if len(s) != len(t):
            return False
        s_dict = {}
        for i in s:
            if i in s_dict:
                s_dict[i] += 1 
            else:
                s_dict[i] = 1
        for j in t:
            if j not in s_dict:
                return False
            s_dict[j] -= 1
        for val in s_dict.values():
            if val != 0:
                return False
        return True
```

### \290. Word Pattern

```python
#12:32
#two hash map
class Solution:
    def wordPattern(self, pattern: str, str: str) -> bool:
        words = str.split(' ')
        if len(pattern) != len(words):
            return False
        word_map = {}
        char_map = {}
        for c,w in zip(pattern, words):
            if c not in char_map:
                if w in word_map:
                    return False
                else:
                    word_map[w] = c
                    char_map[c] = w 
            else:
                if char_map[c] != w:
                    return False
        return True
```

### \966. Vowel Spellchecker

```python
#13:52
#hash table, check three dicts:
'''
if in wordlist: true
if word.lower() in decaplist: true
if word.devowel() in devowlist: true
else: false
'''
class Solution:
    def spellchecker(self, wordlist: List[str], queries: List[str]) -> List[str]:
        #"yellow"--> "y*ll*w"
        def devowel(word):
            return "".join('a' if c in 'aeiou' else c for c in word)

        def check(query):
            if query in dict1:
                return query

            low_query = query.lower()
            if low_query in dict2:
                return dict2[low_query]

            devow_query = devowel(low_query)
            if devow_query in dict3:
                return dict3[devow_query]

            return ""

        #1. creat three dicts
        dict1 = set(wordlist)
        dict2 = {}
        dict3 = {}
        for word in wordlist:
            lower = word.lower()
            dict2.setdefault(lower, word)
            dict3.setdefault(devowel(lower), word)
        print(dict1, dict2, dict3)    
        #2. check query in three dicts
        ans = []
        for query in queries:
            ans.append(check(query))
        return ans
```

## Heap

### 原理

完全二叉树

add\(element\): logN--&gt;sift up

pop\(\) :logN --&gt;sift down

Get min\( \): 1

delete/remove\(element\): N //need identical--&gt;sift up **or** down

​ How to get to O\(logN\): heap + hash map

Minheap: parent &lt;= kids

实现：dynamic array动态数组

array\[0\]: numbers

i's parent == i//2

i's kids == 2i, 2i+1

if full, expend length

* 应用:优先队列 Priority Queue
* 替代品:TreeMap

支持操作:O\(log N\) Add / O\(log N\) Remove / O\(1\) Min or Max Max Heap vs Min Heap

### [Heapify](https://www.lintcode.com/problem/heapify/description)

Given an integer array, heapify it into a min-heap array.

For a heap array A, A\[0\] is the root of heap, and for each A\[i\], A\[i  _2 + 1\] is the left child of A\[i\] and A\[i_  2 + 2\] is the right child of A\[i\].

```python
'''
al: sift down
time:O(n)
begin from n//2 to 0, at most n/4 nums sift 1 time, n/8 sift 2 times,
...1 sift logn times. 
Thus, n/4+n/8*2+...+1*logn = N
'''
class Solution:
    """
    @param: A: Given an integer array
    @return: nothing
    """
    def heapify(self, A):
        # write your code here
        if not A:
            return A
        for i in range(len(A)//2-1, -1, -1):
            self.siftdown(A,i)
        return A
    
    def siftdown(self, A, i):
        n = len(A)
        while i < n:
            now = i 
            left = 2*now+1 
            right = 2*now+2 
            if left < n and A[left] < A[now]:
                now = left
            if right < n and A[right] < A[now]:
                now = right
                
            if now == i:
                break
            A[now], A[i] = A[i], A[now]
            i = now 
            
            
```

递归版本，siftup and siftdown

```python
class Solution:
    """
    @param: A: Given an integer array
    @return: nothing
    """
    def heapify(self, A):
        # write your code here
        for i in range(len(A)):
            self.siftup(A,i)
        for i in range(len(A)//2-1, -1, -1):
            self.siftdown(A,i)
        return A
        
    #time:O(nlogn), same as insert
    def siftup(self, A, i):
        if i <= 0:
            return 
        parent = (i-1)//2
        if A[parent] <= A[i]:
            return
        A[parent], A[i] = A[i], A[parent]
        self.siftup(A, parent)
        
    
    #time:O(N)
    def siftdown(self, A, i):
        n = len(A)
        if 2*i+1 >= n:
            return
        now = i 
        left = 2*now+1 
        right = 2*now+2 
        if left < n and A[left] < A[now]:
            now = left
        if right < n and A[right] < A[now]:
            now = right
            
        if now == i:
            return
        A[now], A[i] = A[i], A[now]
        self.siftdown(A,now)
            
            
```

### \264. Ugly Number II

```python
'''
version1: hashset + pq
time:O(nlogn) space:O(n)
thought:
    1
    add(1*2, 1*3, 1*5), min, popmin -> pq
    unique -> hashset
'''

import heapq
class Solution:
    def nthUglyNumber(self, n: int) -> int:
        heap = [1]
        seen = set([1])
        for i in range(n):
            #get i-th ugly number
            cur_ugly = heapq.heappop(heap)
            #add new 3
            for a in [2, 3, 5]:
                new = cur_ugly*a
                if new not in seen:
                    heapq.heappush(heap, new)
                    seen.add(new)
        return cur_ugly



'''
#version2: dynamic programming
time:O(n) space:O(n)
thought:
    only push next min ugly num, three points to decide min(l[p2]*2,l[p3]*3,l[p5]*5)
'''
class Solution:
    def nthUglyNumber(self, n: int) -> int:
        p1, p2, p3 =0,0,0
        dp = [0]*n
        dp[0] = 1 
        for i in range(1,n):
            new = min(dp[p1]*2, dp[p2]*3, dp[p3]*5)
            dp[i] = new
            if new == dp[p1]*2:
                p1 += 1
            if new == dp[p2]*3:  
                p2 += 1 
            if new == dp[p3]*5:
                p3 += 1
        return dp[n-1]
```

### \263. Ugly Number

```python
#10:23
class Solution:
    def isUgly(self, num: int) -> bool:
        while num >1 and num%2 == 0:
            num = num/2
        while num >1 and num%3 == 0:
            num = num/3
        while num >1 and num%5 == 0:
            num = num/5
        if num == 1:
            return True
        return False

#10:23
class Solution:
    def isUgly(self, num: int) -> bool:
        for p in 2, 3, 5:
            while num % p == 0 < num:
                num /= p
        return num == 1
```

### \1201. Ugly Number III

```python
'''
binary search
time:O(log2*10^9) space:O(1)
count(A) = number of integers<=A which divisible by a or b or c
         = A//a+A//b+A//c-A//ab-A//ac-A//bc+A//abc 
'''
class Solution:
    def nthUglyNumber(self, n: int, a: int, b: int, c: int) -> int:
        ab, bc, ac = self.lcm(a, b), self.lcm(b, c), self.lcm(c, a)
        abc = self.lcm(ab, c)

        s, e = 1, 2*10**9       #^用**表示
        while s+1 < e:
            mid = s + (e - s)//2
            if self.count(mid, a, b, c, ab, bc, ac, abc) < n:
                s = mid
            else:
                e = mid
        if self.count(s, a, b, c, ab, bc, ac, abc) >=n:
            return s
        return e

    def lcm(self, x, y):
            return x * y // math.gcd(x, y) #math.gcd:最大公约数

    def count(self, A, a, b, c,ab, bc, ac, abc):
        return A//a+A//b+A//c-A//ab-A//ac-A//bc+A//abc
```

### \313. Super Ugly Number

```python
#16:35
#three points + dynamic programming
class Solution:
    def nthSuperUglyNumber(self, n: int, primes: List[int]) -> int:
        if n == 1:
            return 1
        points = [0]*len(primes)
        dp = [0]*n
        dp[0] = 1
        for i in range(1,n):
            news = self.creat_uglys(primes, points, dp)
            dp[i] = min(news)
            self.update_points(dp[i], points, news)
        return dp[n-1]

    def creat_uglys(self, primes, points, dp):
        creat = []
        for i in range(len(primes)):
            creat.append(primes[i]*dp[points[i]])
        return creat

    def update_points(self, min_new, points, news):
        for i in range(len(news)):
            if min_new == news[i]:
                points[i] += 1
```

### 545. Top k Largest Numbers II

link code:

mplement a data structure, provide two interfaces:

1. `add(number)`. Add a new number in the data structure.
2. `topk()`. Return the top _k_ largest numbers in this data structure. _k_ is given when we create the data structure.

```python
'''
minheap: 
add:check min
topk: reverse and output

space:O(k)

'''
import heapq
class Solution:
    """
    @param: k: An integer
    """
    def __init__(self, k):
        # do intialization if necessary
        self.heap = []
        self.cap = k 

    """
    @param: num: Number to be added
    @return: nothing
    time:O(logk)
    """
    def add(self, num):
        # write your code here
        heapq.heappush(self.heap, num)
        if len(self.heap) > self.cap:
            heapq.heappop(self.heap)

    """
    @return: Top k element
    time:O(klogk)
    """
    def topk(self):
        # write your code here
        return sorted(self.heap, reverse = True)
```

### 🌟\23. Merge k Sorted Lists

**三种不同的解法**：都要掌握

mergeHelper\_v1\_minHeap 小顶堆（优先队列） mergeHelper\_v2\_Divide\_Conquer 分治思想，递归 mergeHelper\_v3\_Non\_Recursive 两两合并，非递归

时间复杂度均为O\(nlogk\)

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next

#V1: minheap
#使用了self.counter，使得每次heappush都会有一个唯一的值，并将这个值放入tuple, (node.val, counter, node)，这样永远也不会比较node

import heapq
class Solution:
    def mergeKLists(self, lists: List[ListNode]) -> ListNode:

        def new_push(node):
            heapq.heappush(self.heap, (node.val, self.counter, node))
            self.counter += 1

        if not lists:
            return None
        self.heap = []
        self.counter = 0
        #1. build minheap
        for node in lists:
            if node:
                new_push(node)

        #2. heappop
        dummy = ListNode(0)
        head = dummy
        while self.heap:
            new = heapq.heappop(self.heap)[2]
            head.next = new
            head = head.next
            if new.next:
                new_push(new.next)
        return dummy.next
```

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next

#V2: divide and conquer
class Solution:
    def mergeKLists(self, lists: List[ListNode]) -> ListNode:
        if not lists:
            return None
        return self.merge_range_lists(lists, 0, len(lists)-1)

    def merge_range_lists(self, lists, start, end) -> ListNode:
        if start == end:
            return lists[start]

        mid = (start+end)//2
        left = self.merge_range_lists(lists, start, mid)
        right = self.merge_range_lists(lists, mid+1, end)
        return self.merge_two(left, right)

    def merge_two(self, left, right):
        dummy = ListNode(0)
        tail = dummy
        while left and right:
            if left.val <right.val:
                tail.next = left
                left = left.next
            else:
                tail.next = right
                right = right.next
            tail = tail.next
        if left:
            tail.next = left
        elif right:
            tail.next = right
        return dummy.next
```

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next

#V3: 2_way merge 
class Solution:
    def mergeKLists(self, lists: List[ListNode]) -> ListNode:
        if not lists:
            return None
        while len(lists)>1:
            new_lists = []
            for i in range(0, len(lists), 2):
                if i+1 < len(lists):
                    new_one = self.merge_two(lists[i], lists[i+1])
                else:
                    new_one = lists[i]
                new_lists.append(new_one)
            lists = new_lists
        return lists[0]

    def merge_two(self, left, right):
        dummy = ListNode(0)
        tail = dummy
        while left and right:
            if left.val <right.val:
                tail.next = left
                left = left.next
            else:
                tail.next = right
                right = right.next
            tail = tail.next
        if left:
            tail.next = left
        elif right:
            tail.next = right
        return dummy.next
```

similar problems:

### 486. Merge K Sorted Arrays

lincode:

Given _k_ sorted integer arrays, merge them into one sorted array.

Example

```text
Input: 
  [
    [1, 3, 5, 7],
    [2, 4, 6],
    [0, 8, 9, 10, 11]
  ]
Output: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
```

```python
import heapq
class Solution:
    """
    @param arrays: k sorted integer arrays
    @return: a sorted array
    """
    #min heap
    def mergekSortedArrays(self, arrays):
        # write your code here
        if not arrays:
            return None
        heap = []
        #build k minheap
        for i in range(len(arrays)):
            if arrays[i]:
                heapq.heappush(heap, (arrays[i].pop(0), i))

        ans = []
        while heap:
            data, index = heapq.heappop(heap)
            ans.append(data)
            if arrays[index]:
                heapq.heappush(heap, (arrays[index].pop(0), index))
        return ans






class Solution:
    """
    @param arrays: k sorted integer arrays
    @return: a sorted array
    """
    def mergekSortedArrays(self, arrays):
        # write your code here
        #V2: divide and conquer
        if not arrays:
            return None 
        return self.merge_range(arrays, 0, len(arrays)-1)

    def merge_range(self, arrays, start, end):
        if start == end:
            return arrays[start]

        mid = (start + end) // 2 
        left = self.merge_range(arrays, start, mid)
        right = self.merge_range(arrays, mid+1, end)

        return self.merge_two(left, right)

    def merge_two(self, left, right):
        new = []
        l, r = 0, 0 
        while l<len(left) and r<len(right):
            if left[l] < right[r]:
                new.append(left[l])
                l+=1 
            else:
                new.append(right[r])
                r+= 1 

        while l<len(left):
            new.append(left[l])
            l+=1
        while r<len(right):
            new.append(right[r])
            r+=1
        return new
```

### 577. Merge K Sorted Interval Lists

```python
"""
Definition of Interval.
class Interval(object):
    def __init__(self, start, end):
        self.start = start
        self.end = end
"""
#V3 merge 2way
class Solution:
    """
    @param intervals: the given k sorted interval lists
    @return:  the new sorted interval list
    """
    def mergeKSortedIntervalLists(self, intervals):
        # write your code here
        if not intervals:
            return None 
        while len(intervals)>1:
            new_intervals = []
            for i in range(0, len(intervals), 2):
                if i+1 < len(intervals):
                    new = self.merge_inters(intervals[i], intervals[i+1])
                else:
                    new = intervals[i]
                new_intervals.append(new)
            intervals = new_intervals
        return intervals[0]

    def merge_inters(self, left, right):
        l, r = 0, 0 
        ans = []
        while l < len(left) and r < len(right):
            if left[l].start < right[r].start:
                self.merge_two_inter(ans, left[l])
                l += 1 
            else:
                self.merge_two_inter(ans, right[r])
                r += 1 
        while l < len(left) :
            self.merge_two_inter(ans, left[l])
            l += 1 

        while r < len(right):
            self.merge_two_inter(ans, right[r])
            r += 1    
        return ans 

    def merge_two_inter(self, ans, inter):
        if not ans:
            ans.append(inter)
        elif ans[-1].end < inter.start:
            ans.append(inter)
        else:
            ans[-1].end = max(ans[-1].end, inter.end)
```

## **🌟Three solutions to this K-th problem.**

### 973. K Closest Points to Origin

```python
#sort
#time: O(nlogn) space: O(1)
class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        points.sort(key = lambda P: P[0]**2 + P[1]**2)
        return points[:K]


#heap
#time: o(nlogk) space:o(k)
import heapq
class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        if not points or len(points) <= K:
            return points
        heap = []
        while points:
            x,y = points.pop(0)
            heapq.heappush(heap, (-x*x-y*y, x, y))
            if len(heap) > K:
                heapq.heappop(heap)
        ans = [[x,y] for z,x,y in heap]
        return ans
        
            
#quick sort
##time: o(n) space:o(1)
class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        self.quickFindKth(points, 0, len(points)-1, K)
        return points[:K]
    
    #find kth cloest points
    def quickFindKth(self, nums, start, end, k):
        s, e = start, end
        if s == e:
            return 
        pivot = nums[(s+e)//2]
        while s <= e:
            while s <= e and self.compare(nums[s], pivot) < 0:
                s += 1
            while s <= e and self.compare(nums[e], pivot) > 0:
                e -= 1    
            if s <= e:
                nums[s], nums[e] = nums[e], nums[s]
                s += 1
                e -= 1
        if start + k - 1 <= e:
            self.quickFindKth(nums, start, e, k)
        elif start + k - 1 >= s:
            self.quickFindKth(nums, s, end, k-s+start)
        else:
            return
    
    def compare(self, p1, p2):
        return p1[0] * p1[0] + p1[1] * p1[1] - p2[0] * p2[0] - p2[1] * p2[1]
            
```

## 

