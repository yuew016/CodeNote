# 2.2 Linked List & Array

## 5.1 Linked List

• Dummy Node

• High Frequency •

### reverse linked list

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
    @param head: n
    @return: The new head of reversed linked list.
    """
    def reverse(self, head):
        # write your code here
        pre = None
        while head!= None:
            temp = head.next
            head.next = pre 
            pre = head 
            head = temp 
        return pre
```

### Reverse Linked List II

中文English

Reverse a linked list from position m to n.

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
    @param head: ListNode head is the head of the linked list 
    @param m: An integer
    @param n: An integer
    @return: The head of the reversed ListNode
    """
    def reverseBetween(self, head, m, n):
        # write your code here
        dummy = ListNode(-1, head)
        pre_mth = self.findkth(dummy, m-1)
        mth = pre_mth.next 
        nth = self.findkth(dummy, n)
        nth_next = nth.next
        nth.next = None 

        self.reverse(mth)

        pre_mth.next = nth 
        mth.next = nth_next

        return dummy.next 


    def findkth(self, head, k):
        for i in range(k):
            if not head:
                return None 
            head = head.next 
        return head 

    def reverse(self, head):
        pre = None
        while head:
            temp = head.next 
            head.next = pre 
            pre = head 
            head = temp
        return pre
```

### Reverse Nodes in k-Groups

模拟法

Dummy Node:

链表结构发生变化时,就需要 Dummy Node

```python
newhead=ListNode(0)
newhead.next=start

...

return nhead.next
```

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
    @param head: a ListNode
    @param k: An integer
    @return: a ListNode
    """
    def reverseKGroup(self, head, k):
        # write your code here
        dummy = ListNode(0)
        dummy.next = head 
        head = dummy
        while head:
            head = self.reverseK(head, k)

        return dummy.next 
    # head -> n1 -> n2 ... nk -> nk+1
    # =>
    # head -> nk -> nk-1 .. n1 -> nk+1
    # return n1
    def reverseK(self, head, k):
        nk = head 
        for i in range(k):
            nk = nk.next
            if not nk:
                return nk 

        #reverse 
        nk_next = nk.next 
        nk.next = None 
        pre = head 
        first = head.next
        cur = head.next
        while cur:
            temp = cur.next
            cur.next = pre 
            pre = cur 
            cur = temp
        head.next = pre 
        first.next = nk_next
        return first
```

### Partition List

中文English

Given a linked list and a value x, partition it such that all nodes less than x come before nodes greater than or equal to x.

You should preserve the original relative order of the nodes in each of the two partitions.

双指针方法，用两个指针将两个部分分别串起来。最后在将两个部分拼接起来。 left指针用来串起来所有小于x的结点， right指针用来串起来所有大于等于x的结点。 得到两个链表，一个是小于x的，一个是大于等于x的，做一个拼接即可。

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
    @param head: The first node of linked list
    @param x: An integer
    @return: A ListNode
    """
    #双指针法
    def partition(self, head, x):
        # write your code here
        if head is None:
            return head
        shead, bhead = ListNode(0), ListNode(0)
        stail, btail = shead, bhead
        while head is not None:
            if head.val < x:
                stail.next = head
                stail = stail.next 
            else:
                btail.next = head 
                btail = btail.next 
            head = head.next    

        stail.next = bhead.next
        btail.next = None

        return shead.next
```

### Merge Two Sorted Lists

中文English

Merge two sorted \(ascending\) linked lists and return it as a new sorted list. The new sorted list should be made by splicing together the nodes of the two lists and sorted in ascending order.

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
    @param l1: ListNode l1 is the head of the linked list
    @param l2: ListNode l2 is the head of the linked list
    @return: ListNode head of linked list
    """
    def mergeTwoLists(self, l1, l2):
        # write your code here
        if l1 is None:
            return l2
        elif l2 is None:
            return l1 

        dummy = ListNode(0)
        head = dummy
        while l1 and l2:
            if l1.val < l2.val:
                head.next = l1 
                l1 = l1.next
            else:
                head.next = l2
                l2 = l2.next
            head = head.next        
        if l1:
            head.next = l1 
        elif l2:
            head.next = l2 

        return dummy.next
```

### Swap Two Nodes in Linked List

中文English

Given a linked list and two values `v1` and `v2`. Swap the two nodes in the linked list with values `v1` and `v2`. It's guaranteed there is no duplicate values in the linked list. If v1 or v2 does not exist in the given linked list, do nothing.

Example

**Example 1:**

```text
Input: 1->2->3->4->null, v1 = 2, v2 = 4
Output: 1->4->3->2->null
```

**Example 2:**

```text
Input: 1->null, v1 = 2, v2 = 1
Output: 1->n
```

Notice

You should swap the two nodes with values `v1` and `v2`. Do not directly swap the values of the two nodes.

找到权值为 `v1` 和 `v2` 的节点之后, 分情况讨论:

* 如果二者本身是相邻的, 则一共需要修改三条边\(即三个next关系\) {a node} -&gt; {v = v1} -&gt; {v = v2} -&gt; {a node}
* 如果二者是不相邻的, 则一共需要修改四条边 {a node} -&gt; {v = v1} -&gt; {some nodes} -&gt; {v = v2} -&gt; {a node} \(假定v1在v2前\)

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
    @param head: a ListNode
    @param v1: An integer
    @param v2: An integer
    @return: a new head of singly-linked list
    """
    def swapNodes(self, head, v1, v2):
        # write your code here
        if head is None:
            return head 

        dummy = ListNode(0)
        dummy.next = head 
        #find node pre1, pre2  
        pre1, pre2 = self.findpres(dummy, v1, v2)

        #swap 
        if pre1 and pre2 :
            if pre1 == pre2:
                return dummy.next
            elif pre1.next == pre2:
                self.swapone(pre1)
            elif pre2.next == pre1:
                self.swapone(pre2)
            else:
                self.swaptwo(pre1, pre2)
        return dummy.next


    def findpres(self, dummy, v1, v2):
        pre = dummy
        head = pre.next
        pre1, pre2 = None, None
        while head:
            if head.val ==v1:
                pre1 = pre 
            elif head.val == v2:
                pre2 = pre 
            pre = head
            head = head.next 
        return pre1, pre2
    #swap pre1 -> v1 -> v2 ->next2 ->...
    #to pre1 -> v2 ->  v1 ->next2 ->...
    def swapone(self, pre1):
        v1 = pre1.next 
        v2 = v1.next
        next2 = v2.next 

        pre1.next = v2 
        v2.next = v1 
        v1.next = next2

    #swap pre1 -> v1 -> next1 ... pre2 -> v2 ->next2 
    #to pre1 -> v2 -> next1 ... pre2 -> v1 ->next2     
    def swaptwo(self, pre1, pre2):
        v1 = pre1.next
        next1 = v1.next
        v2 = pre2.next
        next2 = v2.next

        pre1.next = v2
        pre2.next = v1
        v2.next = next1
        v1.next = next2
```

### Reorder List

Given a singly linked list L: L0→L1→…→Ln-1→Ln, reorder it to: L0→Ln→L1→Ln-1→L2→Ln-2→…

You must do this in-place without altering the nodes' values.

For example, Given {1,2,3,4}, reorder it to {1,4,2,3}.

先找到中点，然后把后半段倒过来，然后前后交替合并。

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
    @param head: The head of linked list.
    @return: nothing
    """
    #double points
    #O(n)时间 O(1) 空间
    def reorderList(self, head):
        # write your code here
        if not head or not head.next:
            return head 
        first_head = head 

        mid  = self.find_mid(head)
        tail = self.reverse(mid.next)
        mid.next = None  #前半段最后节点为none
        self.insert_two(head, tail)


    def find_mid(self, head):
        slow = head
        fast = head.next
        while fast != None and fast.next != None :
            fast = fast.next.next
            slow = slow.next
        return slow

    def reverse(self, head):
        pre = None
        while head :
            temp = head.next 
            head.next = pre 
            pre = head 
            head = temp
        return pre 

    def insert_two(self,first_head, se_head):
        dummy = ListNode(0)
        index = 0 
        head = dummy
        while first_head and se_head:
            if index % 2 == 0:
                head.next = first_head
                first_head = first_head.next
            else:
                head.next = se_head
                se_head = se_head.next
            head = head.next
            index += 1 
        if se_head:
            head.next = se_head
        else:
            head.next = first_head
        return dummy.next
```

### Rotate List

中文English

Given a list, rotate the list to the right by `k` places, where _k_ is non-negative.

Example

**Example 1:**

```text
Input:1->2->3->4->5  k = 2
Output:4->5->1->2->3
```

**Example 2:**

```text
Input:3->2->1  k = 1
Output:1->3->2
```

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
    @param head: the List
    @param k: rotate to the right k places
    @return: the list after rotation
    """
    def rotateRight(self, head, k):
        # write your code here
        if head is None or head.next is None:
            return head

        new_tail = self.find(head, k)
        head = new_tail.next 
        new_tail.next = None
        return head

    def find(self, head, k):
        #find tail 
        cur = head 
        length = 1 
        while cur.next:
            cur = cur.next 
            length += 1 
        cur.next = head 
        for i in range(length - k%length - 1):
            head = head.next
        return head
```

### Copy List with Random Pointer

中文English

A linked list is given such that each node contains an additional random pointer which could point to any node in the list or null.

Return a deep copy of the list.

**HashMap** **version**

```python
"""
Definition for singly-linked list with a random pointer.
class RandomListNode:
    def __init__(self, x):
        self.label = x
        self.next = None
        self.random = None
"""

#HashMap version
class Solution:
    # @param head: A RandomListNode
    # @return: A RandomListNode
    def copyRandomList(self, head):
        # write your code here
        if head is None:
            return head 

        HashMap = {}
        newHead = RandomListNode(head.label)
        HashMap[head] = newHead
        p = head 
        q = newHead
        #copy Next_link 
        while p.next:
            q.next = RandomListNode(p.next.label)
            HashMap[p.next] = q.next
            q = q.next
            p = p.next
        #copy random_link
        p = head
        while p:
            if p.random:
                HashMap[p].random = HashMap[p.random]
            p = p.next 

        return newHead
```

Challenge

Could you solve it with O\(1\) space?

```python
"""
Definition for singly-linked list with a random pointer.
class RandomListNode:
    def __init__(self, x):
        self.label = x
        self.next = None
        self.random = None
"""


'''
第一遍扫的时候巧妙运用next指针， 开始数组是1->2->3->4  。 然后扫描过程中 先建立copy节点 1->1`->2->2`->3->3`->4->4`, 
然后第二遍copy的时候去建立边的copy，
拆分节点, 一边扫描一边拆成两个链表，这里用到两个dummy node。第一个链表变回  1->2->3 , 然后第二变成 1`->2`->3`  
'''
class Solution:
    # @param head: A RandomListNode
    # @return: A RandomListNode
    def copyRandomList(self, head):
        # write your code here
        if not head:
            return head

        self.copyNext(head)
        self.copyRandom(head)
        return self.splitList(head)

    def copyNext(self, head):
        while head:
            node = RandomListNode(head.label)
            temp = head.next
            head.next = node 
            node.next = temp 
            head = temp

    def copyRandom(self, head):
        while head:
            if head.random:
                head.next.random = head.random.next
            head = head.next.next

    def splitList(self, head):
        newHead = head.next
        while head:
            temp = head.next
            head.next = temp.next 
            head = head.next 
            if temp.next is not None:
                temp.next = temp.next.next
        return newHead
```

### 141 Linked List Cycle

快慢指针的经典题。 快指针每次走两步，慢指针一次走一步。 在慢指针进入环之后，快慢指针之间的距离每次缩小1，所以最终能相遇。

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def hasCycle(self, head: ListNode) -> bool:
        if not head or not head.next:
            return False
        slow = head
        fast = head.next
        while slow != fast:
            if not fast or not fast.next: 
                return False
            slow, fast = slow.next, fast.next.next             
        return True
```

### 142 Linked List Cycle II

Intersection of Two Linked Lists中文

```python
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def detectCycle(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        if not head or not head.next:
            return None
        slow = fast = node = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if slow == fast:
                break
        else:
            return None   # when there is no cycle
        #if cycle exist
        while node != slow:
            node = node.next
            slow = slow.next
        return node
```

### Sort List

在 O\(_n_ log _n_\) 时间复杂度和常数级的空间复杂度下给链表排序。

merge sort&& fast sort

快排 -- 先找Pivot ，再把链表一分为三，分别是由小于，等于，和大于Pivot节点构成。把大于和小于pivot节点构成的两个链表排好（递归）。 最后把三个链表按照从小到大串成一个链表。

解法2：归并 -- 把链表分成左右两半。 分别排好。最后吧两个排好的链表合并。 时间复杂度没什么好说，都是O\(nlogn\)-平均值。 快排最坏O（n2）。 空间复杂度 都是O\(1\) 的。 由于链表的归并排序不用创建一个长度为N的list。

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next

#version 1: merge sort
class Solution:
    def sortList(self, head: ListNode) -> ListNode:
        if not head or not head.next:
            return head
        #s1:find middle point
        mid = self.find_mid(head)
        #s2: sort left list and right list
        right = self.sortList(mid.next)
        mid.next = None
        left = self.sortList(head)
        #s3: merge sorted left and right
        return self.merge(right, left)

    #快慢指针法 找中点
    def find_mid(self, head) -> ListNode:
        if not head or not head.next:
            return head
        slow, fast = head, head.next
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        return slow

    def merge(self,right, left) -> ListNode:
        if not right:
            return left
        elif not left:
            return right
        dummy = ListNode(0)
        head = dummy
        while right and left:
            if right.val < left.val:
                head.next = right
                right = right.next
            else:
                head.next = left
                left = left.next
            head = head.next
        if right:
            head.next = right
        elif left:
            head.next = left
        return dummy.next
```

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    #version2: quick sort
    def sortList(self, head: ListNode) -> ListNode:
        if not head or not head.next:
            return head

        #s1: find mid pivot
        mid = self.findMid(head)   
        #s2: move less node to pivot's left;move larger node to pivot's right
        leftDummy, rightDummy, midDummy = ListNode(0), ListNode(0), ListNode(0)
        leftTail, rightTail, midTail = leftDummy, rightDummy, midDummy
        while head:
            if head.val< mid.val:
                leftTail.next = head
                leftTail = leftTail.next
            elif head.val > mid.val:
                rightTail.next = head
                rightTail = rightTail.next
            else:
                midTail.next = head
                midTail = midTail.next
            head = head.next   
        #key!!!
        leftTail.next, rightTail.next, midTail.next = None, None, None
        sortedLeft = self.sortList(leftDummy.next)
        sortedRight = self.sortList(rightDummy.next)
        #s3:connect three parts
        return self.connect(sortedLeft, sortedRight, midDummy.next)

    def findMid(self, head):
        if not head or not head.next:
            return head
        slow, fast = head, head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        return slow
    def connect(self, left, right, mid):
        dummy = ListNode(0, left)
        cur = dummy
        while cur.next:
            cur = cur.next
        cur.next = mid
        while cur.next:
            cur = cur.next
        cur.next = right
        return dummy.next
```

### Convert Sorted List to Binary Search Tree

Dfs: 高度平衡，所以将链表从中间分开，左边是左子树，右边是右子树。

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    #set mid node as root
    def sortedListToBST(self, head: ListNode) -> TreeNode:
        if not head:
            return None
        elif not head.next:
            return TreeNode(head.val)
        mid_pre = self.find_mid_pre(head)
        mid = mid_pre.next
        mid_pre.next = None

        root = TreeNode(mid.val)
        root.left = self.sortedListToBST(head)
        root.right = self.sortedListToBST(mid.next)
        return root


    def find_mid_pre(self, head):
        if not head or not head.next:
            return head
        slow, fast = head, head.next
        while fast.next and fast.next.next:
            slow, fast = slow.next, fast.next.next
        return slow
```

### Delete Node in a Linked List

237

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def deleteNode(self, node):
        """
        :type node: ListNode
        :rtype: void Do not return anything, modify node in-place instead.
        """
        node.val = node.next.val
        node.next = node.next.next
```

## 5.2 Array

• Subarray • Sorted Array

## Merge Sorted Array:

### 88. Merge Sorted Array

* 双指针
* 倒序合并

Note: You may assume that A has enough space \(size that is greater or equal to m + n\) to hold additional elements from B. The number of elements initialized in A and B are m and n respectively. 分析:涉及两个有序数组合并,设置i和j双指针,分别从两个数组的尾部想头部移动,并判断A[i](https://www.jiuzhang.com/solutions/merge-sorted-array/)和B[j](https://www.jiuzhang.com/solutions/merge-sorted-array/)的大小关系,从而保证最终数组有序,同时每次index从尾部向头部移动。

```python
class Solution:
    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        """
        Do not return anything, modify nums1 in-place instead.
        """
        pos = m + n - 1
        i = m - 1
        j = n - 1
        while i >= 0 and j >= 0 :
            if nums1[i] > nums2[j]:
                nums1[pos] = nums1[i]
                i -= 1
            else:
                nums1[pos] = nums2[j]
                j -= 1
            pos -= 1

        while j>=0:
            nums1[pos] = nums2[j]
            pos -= 1
            j -= 1
```

* 时间复杂度：

  O\(n+m\)

  ，n,m分别为A,B数组的元素个数

  * 利用双指针各自遍历一遍对应的数组；

* 空间复杂度：

  O\(1\)

  * 只需要新建pointApointA,pointBpointB和indexindex三个整型变量；

### 349. Intersection of Two Arrays

```python
#V1: Hash Set
#time: O(m+n)
#Space: O(max(n.m))
class Solution:
    def intersection(self, nums1: List[int], nums2: List[int]) -> List[int]:
        set1 = set(nums1)
        set2 = set()
        for num in nums2:
            if num in set1:
                set2.add(num)      
        result = []
        for num in set2:
            result.append(num)
        return result

#V2: sort and merge
#time: O(mlogm + nlogn) //排序算法：O(mlogm + nlogn)，双指针扫描算法：O(m+n)
#Space: O(1)
class Solution:
    def intersection(self, nums1: List[int], nums2: List[int]) -> List[int]:
        nums1.sort()
        nums2.sort()
        #双指针遍历
        res = []
        i,j = 0,0
        while i < len(nums1) and j < len(nums2):
            if nums1[i] < nums2[j]:
                i+=1
            elif nums1[i] > nums2[j]:
                j+=1
            else:
                res.append(nums1[i])
                i += 1
                j += 1
                while i < len(nums1) and nums1[i] == nums1[i - 1]:
                    i += 1
                while j < len(nums2) and nums2[j] == nums2[j - 1]:
                    j += 1
        return res
```

### 350 Intersection of Two Arrays II

```python
class Solution:
    def intersect(self, nums1: List[int], nums2: List[int]) -> List[int]:
        nums1.sort()
        nums2.sort()
        i,j = 0, 0
        res = []
        while i < len(nums1) and j < len(nums2):
            if nums1[i] > nums2[j]:
                j += 1
            elif nums1[i] < nums2[j]:
                i += 1
            else:
                res.append(nums1[i])
                j+=1
                i += 1
        return res
```

### 4. Median of Two Sorted Arrays

```python
#二分法
class Solution:
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        n = len(nums1) + len(nums2)
        if n%2 == 0:
            smaller = self.findkth(n//2,nums1,0,nums2, 0)
            bigger = self.findkth(n//2 + 1, nums1, 0, nums2, 0)
            return (smaller + bigger) / 2 
        else:
            return self.findkth(n//2+1, nums1, 0 ,nums2, 0)

    def findkth(self, k, nums1, index1, nums2, index2) :
        #递归的出口
        if index1 == len(nums1):
            return nums2[index2 + k - 1]
        elif index2 == len(nums2):
            return nums1[index1 + k - 1]
        elif k == 1:
            return min(nums1[index1], nums2[index2])

        #比较第k//2个数的大小，扔较小的前k//2个数
        a, b = sys.maxsize, sys.maxsize
        if index1 + k//2  <= len(nums1):
            a = nums1[index1 + k//2 - 1]
        if index2 + k//2  <= len(nums2):
            b = nums2[index2 + k//2 - 1]
        if a < b:
            return self.findkth(k-k//2, nums1, index1+k//2, nums2, index2)
        return self.findkth(k-k//2, nums1, index1, nums2, index2 + k//2)
```

## 5.3 Subarray\(prefix sum\)

### Maximum Subarray

T1: prefix sum：

令 PrefixSum\[i\] = A\[0\] + A\[1\] + ... A\[i - 1\], PrefixSum\[0\] = 0

易知构造 PrefixSum 耗费 O\(n\) 时间和 O\(n\) 空间

如需计算子数组从下标i到下标j之间的所有数之和 则有

**Sum\(i~j\) = PrefixSum\[j + 1\] - PrefixSum\[i\]**

```python
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        #version1: greedy
        #max_sum记录全局最大值，cur_sum记录区间和，如果当前sum>0，那么可以继续和后面的数求和，否则就从0开始
        if not nums or not len(nums):
            return 0
        max_sum, cur_sum = -sys.maxsize, 0
        for num in nums:
            cur_sum += num
            max_sum = max(max_sum, cur_sum)
            cur_sum = max(cur_sum, 0)
        return max_sum




class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        #version2: prefix sum
        if not nums or not len(nums):
            return 0
        result, sum, minSum = -sys.maxsize, 0, 0
        for i in nums:
            #result记录全局最大值，sum记录前i个数的和，minSum记录前i个数中0-i的最小值
            sum += i
            result = max(result, sum - minSum)
            minSum = min(minSum, sum)
        return result
```

Minimum Subarray: 乘-1变max

### Subarray Sum

某一段[l, r](https://www.jiuzhang.com/solutions/subarray-sum/)的和为0， 则其对应presum[l-1](https://www.jiuzhang.com/solutions/subarray-sum/) = presum[r](https://www.jiuzhang.com/solutions/subarray-sum/). presum 为数组前缀和。只要保存每个前缀和，找是否有相同的前缀和即可。

复杂度分析

* 时间复杂度：O\(n\)，n为整数数组的长度
* 空间复杂度：O\(n\)，n为整数数组的长度
* 需使用hash表保存前缀和；

```python
class Solution:
    """
    @param nums: A list of integers
    @return: A list of integers includes the index of the first number and the index of the last number
    """
    def subarraySum(self, nums):
        # write your code here
        prefix_hash = {0:-1}
        pre_sum = 0 
        for i,num in enumerate(nums):
            pre_sum += num 
            if pre_sum in prefix_hash:
                return [prefix_hash[pre_sum]+1, i]
            prefix_hash[pre_sum] = i 
        return [-1, -1]
```

follow up:

### Subarray Sum Closest

Time:O\(nlogn\)

Space:O\(n\)

```python
class Solution:
    """
    @param: nums: A list of integers
    @return: A list of integers includes the index of the first number and the index of the last number
    """
    def subarraySumClosest(self, nums):
        # write your code here
        #prefix sum: O(n)
        prefix_hash = [(0,-1)]
        pre_sum = 0
        for i,num in enumerate(nums):
            prefix_hash.append((prefix_hash[-1][0] + num, i))

        #prefix sort: O(nlogn)     
        prefix_hash.sort()

        #find closest minus in i to i-1 : O(n)
        closest = sys.maxsize 
        answer = []
        for i in range(1,len(prefix_hash)):
            if prefix_hash[i][0] - prefix_hash[i-1][0] < closest:
                closest = prefix_hash[i][0] - prefix_hash[i-1][0] 
                left = min(prefix_hash[i][1], prefix_hash[i-1][1]) + 1 
                right = max(prefix_hash[i][1], prefix_hash[i-1][1])
                answer = [left, right]
        return answer
```

### \1292. Maximum Side Length of a Square with Sum Less than or Equal to Threshold

```python
from itertools import product
class Solution:
    #13:20
    #dp:prefix sum + binary search
    #time:O(mnlog(min(m,n)))
    #space: O(mn)
    def maxSideLength(self, mat: List[List[int]], threshold: int) -> int:
        m,n = len(mat), len(mat[0])
        #prefix sum
        #build prefix sum 
        prefix = [[0]*(n+1) for _ in range(m+1)]
        for i in range(m):
            for j in range(n):
                prefix[i+1][j+1] = prefix[i][j+1] + prefix[i+1][j] - prefix[i][j]+mat[i][j]

        # """reture true if there is such a sub-matrix of length k"""
        ##sum(mat[i:i+k][j:j+k]) = prefix[i+k][j+k] - prefix[i][j+k] - prefix[i+k][j] + prefix[i][j].
        def below(k):
            for i in range(m+1-k):
                for j in range(n+1-k):
                    if prefix[i+k][j+k] - prefix[i][j+k] - prefix[i+k][j] + prefix[i][j] <= threshold:
                        return True
            return False

        #binary search
        lo, hi = 1, min(len(mat), len(mat[0]))
        while lo <= hi: 
            mid = (lo + hi)//2
            if below(mid): lo = mid + 1
            else: hi = mid - 1      
        return hi




from itertools import product
class Solution:
    #13:20
    #dp:prefix sum
    #time:O(mn)
    #space: O(mn)
    def maxSideLength(self, mat: List[List[int]], threshold: int) -> int:
        m,n = len(mat), len(mat[0])

        dp = [[0]*(n+1) for _ in range(m+1)]
        ans = 0
        for r in range(1,m+1):
            for c in range(1,n+1):
                dp[r][c] = dp[r][c-1] + dp[r-1][c] - dp[r-1][c-1]+mat[r-1][c-1]
                k = ans + 1
                if r >= k and c >= k and threshold >= dp[r][c] - dp[r-k][c] - dp[r][c-k] + dp[r-k][c-k]:
                    ans = k             
        return ans
```

### \915. Partition Array into Disjoint Intervals

```python
#17:08
class Solution:
    def partitionDisjoint(self, A):
        N = len(A)
        maxleft = [None] * N
        minright = [None] * N

        m = A[0]
        for i in range(N):
            m = max(m, A[i])
            maxleft[i] = m

        m = A[-1]
        for i in range(N-1, -1, -1):
            m = min(m, A[i])
            minright[i] = m

        for i in range(1, N):
            if maxleft[i-1] <= minright[i]:
                return i
```

## 5.4 More

### 1.Two Sum

```python
#One-pass Hash Table
#time:O(n) space:O(n)
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        hash_map = {}
        for i, num in enumerate(nums):
            diff = target - num
            if diff in hash_map:
                return [hash_map[diff], i]
            hash_map[num] = i
```

### 26 Remove Duplicates from Sorted Array

```python
#two points: i:traversal, to find the next un-same node
#t:tail of new list
#time O(n) space:O(1)
class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        if not nums or len(nums) <= 1:
            return len(nums)
        i,tail = 1, 0
        while i < len(nums):
            if nums[i] != nums[tail]:               
                tail += 1
                nums[tail] = nums[i]
            i+=1
        return tail+1
```

### 27. Remove Element

```python
#two pointers
#time O(n) space:O(1)
class Solution:
    def removeElement(self, nums: List[int], val: int) -> int:
        i, t = 0, 0
        while i < len(nums):
            if nums[i] != val:
                nums[t] = nums[i]
                t += 1
            i += 1
        return t
```

## [Game of Life](https://leetcode.com/problems/game-of-life/)

```python
'''
0,2 are "dead", and "dead->live"
1,3 are "live", and "live->dead"
'''
class Solution:
    def gameOfLife(self, board: List[List[int]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        def alive(x, y, r, c):
            neighbors = [(1,0), (1,-1), (0,-1), (-1,-1), 
                         (-1,0), (-1,1), (0,1), (1,1)]
            count = 0
            for nei in neighbors:
                i = x + nei[0]
                j = y + nei[1]
                if 0<= i < r and 0 <= j < c and board[i][j]%2== 1:
                    count += 1
            return count
    
        
        if not board or not board[0]:
            return
        r, c = len(board), len(board[0])
        for x in range(r):
            for y in range(c):
                if board[x][y] == 0:
                    if alive(x,y,r,c) == 3:
                        board[x][y] = 2
                else:
                    count = alive(x, y, r, c)
                    if count < 2 or count > 3:
                        board[x][y] = 3
        for x in range(r):
            for y in range(c):
                if board[x][y] == 2:
                    board[x][y] = 1
                elif board[x][y] == 3:
                    board[x][y] = 0
        
                
```

follow up: infinite array  
just calculate with live points

```python
import collections
class Solution:
    def gameOfLife(self, board: List[List[int]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        #find all live points
        live = {(i,j) for i, row in enumerate(board) for j, live in enumerate(row) if live}
        #print(live)
        #find all next live points
        live = self.find_next_live(live)
        for i in range(len(board)):
            for j in range(len(board[0])):
                board[i][j] = int((i,j) in live)
    
    def find_next_live(self, live):
        neighbors = collections.Counter()
        #each live add 1 to all its neighbors
        for i,j in live:
            for I in (i-1, i, i+1):
                for J in (j-1, j, j+1):
                    if I != i or J != j:
                        neighbors[I, J] += 1
        new_live = set()
        for ij in neighbors:
            if neighbors[ij] == 3 or neighbors[ij] == 2 and ij in live:
                new_live.add(ij)
        return new_live
```

## [Find Pivot Index](https://leetcode.com/problems/find-pivot-index/)/Balanced Array

```python
#prefix sum
#time:O(n), spaceLO(n)
class Solution:
    def pivotIndex(self, nums: List[int]) -> int:
        #state: p[i] = sum of nums in [0...i]
        presum = [0]*(len(nums)+1)
        for i in range(1,len(presum)):
            presum[i] = presum[i-1] + nums[i-1]
        #print(presum)
        for i in range(1, len(presum)):
            left = presum[i-1] - presum[0]
            right = presum[-1] - presum[i]
            if left == right:
                return i-1
        return -1
  
            
                                
#check if left*2 + num[i] == sum
#time:O(n), spaceLO(1)
class Solution:
    def pivotIndex(self, nums: List[int]) -> int:
        total = sum(nums)
        left = 0
        for i in range(len(nums)):
            if left*2 + nums[i] == total:
                return i
            left += nums[i]
        return -1
        
            
        
 #check if left == right
#time:O(n), space：O(1)
class Solution:
    def pivotIndex(self, nums: List[int]) -> int:
        left, right = 0, sum(nums)
        for i in range(len(nums)):
            right -= nums[i]
            if left == right:
                return i
            left += nums[i]
        return -1
        
```

