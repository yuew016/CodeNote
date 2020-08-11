# Binary Tree & Divide Conquer

二叉树n结点， 高度为\[logN, N\]

只有balanced binary tree/Optimal binary search tree = logn

iterative

背诵两个程序： • 非递归版本的 Pre Order, In Order

## 2.1 时间复杂度训练 II

二叉树的时间复杂度一般都为O\(n\)：节点个数\*每个节点处理时间

通过O\(n\)的时间，把规模为n的问题变为n/2?: O\(n\)

通过O\(1\)的时间，把规模为n的问题变为n/2: O\(logN\)

通过O\(n\)的时间，把n的问题，变为了**两个n/2**的问题，复杂度是多少? O\(NlogN\)

Merge sort, Quick sort

通过O\(1\)的时间，把n的问题，变成了**两个n/2**的问题，复杂度是多少? O\(n\)

O\(1\)+...+O\(n\)=O\(n\)

## 2.2 Templete

递归三要素：preorder

1. 递归的定义
2. 递归的拆减
3. 递归的出口
4. 递归的调用

   递归（遍历，分治），非递归：Recursion \(Traverse, Divide Conquer\), Nonrecursion

### 2.2.1 preorder:根左右

```python
"""
Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None
"""
#Version 1: Traverse
class Solution:
    """
    @param root: A Tree
    @return: Preorder in ArrayList which contains node values.
    """
    def preorderTraversal(self, root):
        # write your code here
        self.results = []
        self.traverse(root)
        return self.results

    def traverse(self,root):
        if not root:
            return
        self.results.append(root.val)
        self.traverse(root.left)
        self.traverse(root.right)


#Version 2: divide and conquer
    def preorderTraversal(self, root) -> []:
        # write your code here
        results = []    
        #null or leaf
        if not root:
            return results
        #divide
        left = self.preorderTraversal(root.left)
        right = self.preorderTraversal(root.right)        
        #conquer
        results.append(root.val)
        results.extend(left)
        results.extend(right)

        return results

#version 3: non-recursion
        def preorderTraversal(self, root):
        # write your code here
        if not root:
            return []
        stack = [root]
        results = []
        while stack:
            node = stack.pop()
            results.append(node.val)
            if node.right:
                stack.append(node.right)
            if node.left:
                stack.append(node.left)

        return results
```

​

### 2.2.2 Inorder: 左根右

```python
"""
Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None
"""

class Solution:
    """
    @param root: A Tree
    @return: Inorder in ArrayList which contains node values.
    """
    #version 1: tranverse
    def inorderTraversal(self, root):
        # write your code here
        self.inorder = []
        self.tranverse(root)
        return self.inorder

    def tranverse(self, root):
        if not root:
            return
        self.tranverse(root.left)
        self.inorder.append(root.val)
        self.tranverse(root.right)

    #version 2: divide and conquer
    def inorderTraversal(self, root) -> []:
        # write your code here
        result = []
        if not root:
            return result

        left = self.inorderTraversal(root.left)
        right = self.inorderTraversal(root.right)

        result.extend(left)
        result.append(root.val)
        result.extend(right)

        return result      


    #version 3: non-recursion
    #添加所有最左边节点到栈。
    #pop stack 然后添加到结果。
    #查找当前node的右边节点是否为空， 如果不为空，重复step 1。
    def inorderTraversal(self, root):
        # write your code here
        if not root:
            return []

        inorder = []
        stack = []

        while root:
            stack.append(root)
            root = root.left

        while stack:
            cur = stack.pop()
            inorder.append(cur.val)

            if cur.right:
                root = cur.right
                while root:
                    stack.append(root)
                    root = root.left

        return inorder
```

### 2.2.3 Postorder：左右根

```python
"""
Definition of TreeNode:
class TreeNode:        
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None
"""

class Solution:
    """
    @param root: A Tree
    @return: Postorder in ArrayList which contains node values.
    """
    #version 1: tranverse
    def postorderTraversal(self, root):
        # write your code here
        self.postorder = []
        self.tranverse(root)
        return self.postorder

    def tranverse(self, root):
        if not root:
            return

        self.tranverse(root.left)
        self.tranverse(root.right)
        self.postorder.append(root.val)


    #version 2: divide and conquer
    def postorderTraversal(self, root) -> []:
        # write your code here
        result = []
        if not root:
            return result

        left = self.postorderTraversal(root.left)    
        right = self.postorderTraversal(root.right)

        result.extend(left)
        result.extend(right)
        result.append(root.val)

        return result


#version 3: non-recursion
    def postorderTraversal(self, root):
        # write your code here
        if not root:
            return []

        result = []
        stack = []
        cur = root
        while cur:
            stack.append(cur)
            if cur.left:
                cur = cur.left
            else:
                cur = cur.right

        while stack:
            cur = stack.pop()
            result.append(cur.val)
            if stack:
                if stack[-1].left == cur:
                    cur = stack[-1].right
                    while cur:
                        stack.append(cur)
                        if cur.left:
                            cur = cur.left
                        else:
                            cur = cur.right

        return result
```

## 2.3 Examples

### Maximum Depth of Binary Tree

```python
"""
Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None
"""

class Solution:
    """
    @param root: The root of binary tree.
    @return: An integer
    """

    #1. divide and conquer
    def maxDepth(self, root):
        # write your code here
        if not root:
            return 0

        return max(self.maxDepth(root.left), self.maxDepth(root.right)) + 1   



    #2. tranverse
    def maxDepth(self, root):
        # write your code here
        if not root:
            return 0

        self.height = 0
        self.helper(root, 1)
        return self.height

    def helper(self, root, curheight):
        if not root:
            return

        self.helper(root.left, curheight + 1 )
        self.helper(root.right, curheight + 1 )

        if curheight > self.height:
            self.height = curheight
```

### find all root-to-leaf path

Divde and conquer

```python
"""
Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None
"""

class Solution:
    """
    @param root: the root of the binary tree
    @return: all root-to-leaf paths
    """
    #1.定义：返回以root为根的所有路径
    def binaryTreePaths(self, root) -> []:
        # write your code here
        paths = []

        #3.出口
        if root is None:
            return paths
        if root.left is None and root.right is None:
            paths.append(str(root.val))
            return paths

        #2. 拆减
        leftpaths = self.binaryTreePaths(root.left)
        rightpaths = self.binaryTreePaths(root.right)

        for path in leftpaths + rightpaths:
            paths.append(str(root.val) + '->' + path)

        return paths
```

### Minimum Subtree

```python
"""
Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None
"""

class Solution:
    """
    @param root: the root of binary tree
    @return: the root of the minimum subtree
    """
    #version1: 使用 Divide Conquer + Traverse 的方法
    #定义：返回和最小子树的根结点
    def findSubtree(self, root):
        # write your code here
        self.min_sum = float('inf')
        self.min_subroot = None
        self.getTreeSum(root)
        return self.min_subroot

    # 得到 root 为根的二叉树的所有节点之和
    # 顺便打个擂台求出 minimum subtree    
    def getTreeSum(self, root) -> int:
        #3.chukou
        if not root:
            return 0

        #2.拆分
        left = self.getTreeSum(root.left)
        right = self.getTreeSum(root.right)
        sum = left + right + root.val

        #打擂
        if sum < self.min_sum:
            self.min_sum = sum
            self.min_subroot = root

        return sum    

    #veision 2: 使用纯 Divide & Conquer 的方法
    #定义：返回和最小子树的根结点
    def findSubtree(self, root):
        # write your code here
        min_root, min_sum, sum = self.helper(root)
        return min_root

    def helper(self, root):
        if not root:
            return 0, sys.maxsize, 0

        left_min_root, left_min_sum, left_sum = self.helper(root.left)
        right_min_root, right_min_sum, right_sum = self.helper(root.right)

        sum = left_sum + right_sum + root.val
        if left_min_sum == min(left_min_sum, right_min_sum, sum):
            return left_min_root, left_min_sum, sum
        elif right_min_sum == min(left_min_sum, right_min_sum, sum):
            return right_min_root, right_min_sum, sum 

        return root, sum, sum
```

### Subtree with Maximum Average

```python
"""
Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None
"""

class Solution:
    """
    @param root: the root of binary tree
    @return: the root of the maximum average of subtree
    """

    def findSubtree2(self, root):
        # write your code here
        #因为需要求平均数，所以要记录每个节点sum, size
        #分治法计算每一颗子树的平均值，打擂台求出最大平均数的子树
        self.max_average = 0
        self.max_root = None

        self.helper(root)
        return self.max_root

    def helper(self, root):
        if not root:
            return 0, 0

        left_size, left_sum = self.helper(root.left)  
        right_size, right_sum = self.helper(root.right) 

        size = left_size + right_size + 1 
        sum = left_sum + right_sum + root.val

        if self.max_root is None or sum / size > self.max_average:
            self.max_average = sum / size
            self.max_root = root 

        return size, sum
```

### Balanced Binary Tree

Divide and conquer: 搜集每个子树的：1高度 2是否平衡

```python
"""
Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None
"""

class Solution:
    """
    @param root: The root of binary tree.
    @return: True if this Binary tree is Balanced, or false.
    """
    def isBalanced(self, root):
        # write your code here
        Balanced, height = self.helper(root)
        return Balanced

    def helper(self, root):
        if not root:
            return True, 0

        l_Balanced, l_height = self.helper(root.left) 
        r_Balanced, r_height = self.helper(root.right) 

        if l_Balanced and r_Balanced and abs(l_height - r_height) <= 1:
            return True, max(l_height, r_height) + 1 
        return False, 0
```

### Lowest Common Ancestor

给定一棵二叉树，找到两个节点的最近公共父节点\(LCA\)。

最近公共祖先是两个节点的公共的祖先节点且具有最大深度。

with parent pointer vs no parent pointer

follow up: LCA II & III

```python
"""
Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        this.val = val
        this.left, this.right = None, None
"""


class Solution:
    """
    @param: root: The root of the binary tree.
    @param: A: A TreeNode
    @param: B: A TreeNode
    @return: Return the LCA of the two nodes.
    """
    #两个节点一定在这棵树上
    def lowestCommonAncestor(self, root, A, B):
        # write your code here
        #if A&&B in root，return lca(A, B)
        #if A/B in root， return A/B
        #if None in root, return None

        if root is None:
            return None
        if root is A or root is B:
            return root

        left = self.lowestCommonAncestor(root.left, A, B)
        right = self.lowestCommonAncestor(root.right, A, B)

        if left is not None and right is not None:
            return root
        elif left is not None:
            return left
        elif right is not None:
            return right
        return None      

      #V3:如果两个节点在这棵树上不存在最近公共祖先，返回 `null` 。
    def lowestCommonAncestor3(self, root, A, B):
        # write your code here
        node, isA, isB = self.helper(root, A, B)
        if isA and isB:
            return node
        return None     

    def helper(self, root, A, B):
        if not root:
            return None, False, False

        left_node, Left_isA, Left_isB = self.helper(root.left, A, B)
        right_node, right_isA, right_isB = self.helper(root.right, A, B)

        a = Left_isA or right_isA or root is A
        b = Left_isB or right_isB or root is B

        if root == A or root == B :
            return root, a, b
        elif left_node and right_node:
            return root, a, b 
        elif left_node:
            return left_node, a, b 
        elif right_node:
            return right_node, a, b 
        return None, a, b    

    def lowestCommonAncestor3(self, root, A, B):
        # write your code here
         # write your code here
        #if A&&B in root，return lca(A, B)
        #if A/B in root， return A/B
        #if None in root, return None

        self.foundA, self.foundB = False, False 

        lca = self.helper(root, A, B)
        if self.foundA and self.foundB:
            return lca 
        return None    

    def helper(self, root, A, B):
        if not root:
            return None

        left = self.helper(root.left, A, B)
        right = self.helper(root.right, A, B)

        if root == A:
            self.foundA = True
        if root == B:
            self.foundB = True

        if root == A or root == B or (left and right):
            return root 
        elif left:
            return left
        elif right:
            return right
        return None
```

## 2.4 Binary Search Tree BST

从定义出发: • 左子树都比根节点小 • 右子树都不小于根节点

• 从效果出发: • 中序遍历 in-order traversal 是“不下降”序列

### Validate Binary Search Tree

一棵BST定义为：

* 节点的左子树中的值要**严格**小于该节点的值。
* 节点的右子树中的值要**严格**大于该节点的值。
* 左右子树也必须是二叉查找树。
* 一个节点的树也是二叉查找树。

  traverse\(Inorder\) vs divide conquer

```python
"""
Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None
"""
#version1: divide and conquer
class Solution:
    """
    @param root: The root of binary tree.
    @return: True if the binary tree is BST, or false
    """
    def isValidBST(self, root):
        # write your code here
        isBST, maxnode, minnode = self.helper(root)
        return isBST

    def helper(self, root):
        if not root:
            return True, None, None

        l_isBST, l_maxnode, l_minnode = self.helper(root.left)
        r_isBST, r_maxnode, r_minnode = self.helper(root.right)

        if l_isBST is False or r_isBST is False:
            return False, None, None
        if l_maxnode is not None and l_maxnode >= root.val \
        or r_minnode is not None and r_minnode <= root.val:
            return False, None, None
        #is BST
        minnode = l_minnode if l_minnode is not None else root.val
        maxnode = r_maxnode if r_maxnode is not None else root.val
        return True, maxnode, minnode

   #version2: inorder tranverse     
         def isValidBST(self, root):
        # write your code here
        self.isBST = True
        self.lastVal = None
        self.helper(root)
        return self.isBST

    def helper(self, root):
        if not root:
            return 

        self.helper(root.left)
        if self.lastVal is not None and self.lastVal >= root.val:
            self.isBST = False
            return
        self.lastVal = root.val
        self.helper(root.right)
```

### Convert Binary Search Tree to Sorted Doubly Linked List

```python
"""
Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None
"""

class Solution:
    """
    @param root: root of a tree
    @return: head node of a doubly linked list
    """
    #VI: 中序遍历
    def treeToDoublyList(self, root):
        # Write your code here.
        if root is None:
            return root 
        self.cur = None
        self.pre = None
        self.inorder(root)
        self.cur.left = self.pre
        self.pre.right = self.cur
        return self.cur

    def inorder(self, root):
        if root is None:
            return
        self.inorder(root.left)

        if self.cur is None:
            self.cur = root 
        if self.pre:
            self.pre.right = root
            root.left =self.pre
        self.pre = root 

        self.inorder(root.right)


    #VII: divide and conquer
    def treeToDoublyList(self, root):
        # Write your code here.
        if not root:
            return root
        head, tail = self.helper(root)
        head.left = tail
        tail.right = head
        return head

    def helper(self, root):
        if not root:
            return None, None
        left_head, left_tail = self.helper(root.left)
        right_head, right_tail = self.helper(root.right)

        if left_tail:
            root.left = left_tail
            left_tail.right = root

        if right_head:
            root.right = right_head
            right_head.left = root 

        head = left_head or root or right_head
        tail = right_tail or root or left_tail
        return head, tail
```

### Flattern Binary Tree to Linked List

```python
"""
Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None
"""

class Solution:
    """
    @param root: a TreeNode, the root of the binary tree
    @return: nothing
    """
    #divide and conquer
    def flatten(self, root):
        # write your code here
        if not root:
            return root
        self.helper(root)
        return root

    def helper(self, root):
        if not root:
            return None 

        left_last = self.helper(root.left)
        right_last = self.helper(root.right)

        if left_last:
            left_last.right = root.right 
            root.right = root.left 
            root.left = None 

        return right_last or left_last or root    



    #VII: transerve
    lastnode = None
    def flatten(self, root):
        # write your code here

        if not root:
            return root
        if self.lastnode is not None:
            self.lastnode.left = None
            self.lastnode.right = root

        self.lastnode = root
        right = root.right
        self.flatten(root.left)
        self.flatten(right)
```

## [Validate IP Address](https://leetcode.com/problems/validate-ip-address/)

```python
class Solution:
    def validate_IPv4(self, IP: str) -> str:
        nums = IP.split('.')
        for x in nums:
            # Validate integer in range (0, 255):
            # 1. length of chunk is between 1 and 3
            if len(x) == 0 or len(x) > 3:
                return "Neither"
            # 2. no extra leading zeros
            # 3. only digits are allowed
            # 4. less than 255
            if x[0] == '0' and len(x) != 1 or not x.isdigit() or int(x) > 255:
                return "Neither"
        return "IPv4"
    
    def validate_IPv6(self, IP: str) -> str:
        nums = IP.split(':')
        hexdigits = '0123456789abcdefABCDEF'
        for x in nums:
            # Validate hexadecimal in range (0, 2**16):
            # 1. at least one and not more than 4 hexdigits in one chunk
            # 2. only hexdigits are allowed: 0-9, a-f, A-F
            if len(x) == 0 or len(x) > 4 or not all(c in hexdigits for c in x):
                return "Neither"
        return "IPv6"
        
    def validIPAddress(self, IP: str) -> str:
        if IP.count('.') == 3:
            return self.validate_IPv4(IP)
        elif IP.count(':') == 7:
            return self.validate_IPv6(IP)
        else:
            return "Neither"
```

