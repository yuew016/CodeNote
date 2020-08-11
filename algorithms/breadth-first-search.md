# Breadth First Search

## 什么时候应该使用BFS?

图的遍历 **Traversal in Graph**

* 层级遍历 Level Order Traversal
* 由点及面 Connected Component ：判断连通性
* 拓扑排序 Topological Sorting

  最短路径 **Shortest Path in Simple Graph** • 仅限简单图求最短路径

  即，图中每条边长度都是1，且没有方向

## 3.1 二叉树上的宽搜 BFS in Binary Tree

Binary Tree Level Order Traversal

```python
#V1: deque: double queue
from collections import deque

class Solution:
    """
    @param root: A Tree
    @return: Level order a list of lists of integer
    """
    def levelOrder(self, root):
        if root is None:
            return []

        queue = deque([root])
        result = []
        while queue:
            level = []
            size = len(queue)
            for _ in range(size): #queue长度变化不会有影响，因为range()是生成一个固定的值
                node = queue.popleft()
                level.append(node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            result.append(level)
        return result
```

## 3.2 Binary Tree Serialization \(M+Y\)

### 什么是序列化?

将“内存”中结构化的数据变成“字符串”的过程

序列化:object to string

反序列化:string to object

### 常用的一些序列化手段:

* XML
* Json
* Thrift \(by Facebook\)
* ProtoBuf \(by Google\)

序列化算法 一些序列化的例子:

* 比如一个数组，里面都是整数，我们可以简单的序列化为”\[1,2,3\]”
* 一个整数链表，我们可以序列化为，”1-&gt;2-&gt;3”
* 一个哈希表\(HashMap\)，我们可以序列化为，”{\”key\”: \”value\”}”

### 二叉树序列化

* 二叉树如何序列化? 你可以使用任何你想要用的方法进行序列化，只要保证能够解析回来即可。

  LintCode 采用的是 BFS 的方式对二叉树数据进行序列化，这样的好处是，你可以更为容易的自己画出 整棵二叉树。

## 3.3 图上的宽搜 BFS in Graph

### Graph Valid Tree

图的遍历\(由点及面\) 条件1:刚好N-1条边 条件2:N个点连通

灌水法

dictionary 建立邻接表，queue做BFS用，visited为hashset，快速查找。 先判断边数是否为节点数-1，再判断是否所有点可以访问到。两个条件可以看是否无环。

```python
def validTree(self, n, edges):
        # write your code here
        if n == 0:
            return False
        if len(edges) != n - 1:
            return False
        #dictionary 建立邻接表
        dict = {}
        for e in edges:
            dict[e[0]] = dict.get(e[0], []) + [e[1]]
            dict[e[1]] = dict.get(e[1], []) + [e[0]]
        #print (dict[e[0]]) 
        queue = [0]  #queue做BFS用
        visited = set([0]) #visited为hashset
        while queue:
            node = queue.pop(0)
            for i in dict.get(node, []): #dict[node]不对，因为可能不存在
                if i in visited:
                    continue
                queue.append(i)
                visited.add(i)
        return len(visited) == n
```

### Clone Graph

```python
def cloneGraph(self, node):
        # write your code here使用宽度优先搜索 BFS 的版本
        #1.node --> nodes
        if node is None:
            return node 
        nodes = self.getnodes(node)    
        #2.copy nodes
        mapping = {}
        for this in nodes:
            mapping[this]=UndirectedGraphNode(this.label)
        #3.copy edges
        for this in nodes:
            new_node = mapping[this]
            for neighbor in this.neighbors:
                new_neighbor = mapping[neighbor]
                new_node.neighbors.append(new_neighbor)
        return mapping[node]

    def getnodes(self, node):
        if node is None:
            return node 
        q = collections.deque([node])
        result = set([node])
        while q:
            this = q.popleft()
            for neighbor in this.neighbors:
                if neighbor not in result:
                    result.add(neighbor)
                    q.append(neighbor)
        return result
```

## 3.4 拓扑排序

### Topological Sorting

```python
    def topSort(self, graph):
        # write your code here
        order = []
        if graph is None:
            return order
        #1.get indegree
        nodes_indegree = self.get_indegree(graph)

        #2.get start nodes
        start_nodes = [node for node in graph if nodes_indegree[node] == 0]
        #3.bfs
        q = collections.deque(start_nodes)
        while q:
            node = q.popleft()
            order.append(node)
            for neighbor in node.neighbors:
                nodes_indegree[neighbor] -= 1 
                if nodes_indegree[neighbor] == 0:
                    q.append(neighbor)

        return order

    def get_indegree(self,graph):
        nodes_indegree = {x:0 for x in graph}
        for node in graph:
            for neighbor in node.neighbors:
                nodes_indegree[neighbor] += 1 
        return nodes_indegree
```

### Course Schedule裸拓扑排序

找出所有拓扑排序：dfs

```python
def canFinish(self, numCourses, prerequisites):
        # write your code here
        if numCourses <= 1:
            return True
        #get indegree
        first_courses = []
        indegree = [0 for x in range(numCourses)]
        next = {i:[] for i in range(numCourses)}

        for i,j in prerequisites:
            indegree[i] += 1 
            next[j].append(i)
        #get start node    
        first_courses = [x for x in range(numCourses) if indegree[x] == 0]
        #bfs
        q = collections.deque(first_courses)
        count = 0
        while q:
            this = q.popleft()
            count += 1 
            for course in next[this]:
                indegree[course] -= 1 
                if indegree[course] == 0:
                    q.append(course)
        return count == numCourses
```

### Sequence Reconstruction

判断是否有且仅有一个能从 `seqs`重构出来的序列，并且这个序列是`org`。

\(判断是否只存在一个拓扑排序的序列 只需要保证队列中一直最多只有1个元素即可\)

```python
class Solution:
    """
    @param org: a permutation of the integers from 1 to n
    @param seqs: a list of sequences
    @return: true if it can be reconstructed only one or false
    """
    def sequenceReconstruction(self, org, seqs):
        # write your code here
        graph = self.build_graph(seqs)
        order = self.top_order(graph)
        return order == org

    def build_graph(self, seqs):
        graph = {}
        for seq in seqs:
            for node in seq:
                if node not in graph:
                    graph[node] = set()
        for seq in seqs:
            for i in range(1,len(seq)):
                graph[seq[i-1]].add(seq[i])
        return graph  

    def top_order(self, graph):
        if not graph or len(graph) == 0:
            return []
        #get indegree
        indegree = {node:0 for node in graph}
        for node in graph:
            for x in graph[node]:
                indegree[x] += 1 
        #get start nodes         
        start = [node for node in indegree if indegree[node] == 0]
        q = collections.deque(start)
        result = []
        #bfs 
        while q:
            if len(q) > 1:
                return [] 
            node = q.popleft()
            result.append(node)
            for neighbor in graph[node]:
                indegree[neighbor] -= 1 
                if indegree[neighbor] == 0:
                    q.append(neighbor)

        return result
```

## 3.5 BFS in Matrix

### 矩阵 vs 图

图 Graph N个点，M条边 M最大是 O\(N^2\) 的级别 图上BFS时间复杂度 = **O\(N + M\) or O\(M\)**

• 说是O\(M\)问题也不大，因为M一般都比N大 所以最坏情况可能是 O\(N^2\)

矩阵 Matrix R行C列 R_C个点，R_C_2 条边\(每个点上下左右4条边，每条边被2个点共享\)。 矩阵中BFS时间复杂度 = \*\*O\(R_  C\)\*\*

### Number of Islands

图的遍历\(由点及面\)

坐标变换数组

int\[\] deltaX = {1,0,0,-1};

int\[\] deltaY = {0,1,-1,0};

问:写出八个方向的坐标变换数组?

int\[\] deltaX = {1,0,0,-1,1,1,-1,-1};

int\[\] deltaY = {0,1,-1,0,1,-1,1,-1};

```python
class Solution:
    """
    @param grid: a boolean 2D matrix
    @return: an integer
    """
    def numIslands(self, grid):
        # write your code here
        if not grid or not grid[0]:
            return 0
        count = 0    
        visited = set()
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == 1 and (i,j) not in visited:
                    self.helper(grid, i, j, visited)
                    count += 1 
        return count            

    def helper(self, grid, x, y, visited):
        q = collections.deque([(x,y)])
        visited.add((x,y))
        while q:
            tx,ty = q.popleft()
            for dx,dy in [(0,1),(0,-1),(1,0),(-1,0)]: #neighbor的四个方向
                this_x = tx + dx
                this_y = ty + dy 
                if not self.border(grid, this_x, this_y,visited):
                    continue
                q.append((this_x,this_y))
                visited.add((this_x,this_y))
   #判断有效点                
    def border(self, grid, x, y, visited):
        tx, ty = len(grid), len(grid[0])
        if 0<=x<tx and 0<=y<ty and (x,y) not in visited and grid[x][y]:
            return True
        return False
```

### Zombie in Matrix

`1` 代表僵尸，`0` 代表人类\(数字 0, 1, 2\)。僵尸每天可以将上下左右最接近的人类感染成僵尸，但不能穿墙。将所有人类感染为僵尸需要多久，如果不能感染所有人则返回 `-1`。

```python
class Solution:
    """
    @param grid: a 2D integer grid
    @return: an integer
    """
    def zombie(self, grid):
        # write your code here
        if not grid or not grid[0]:
            return 0
        q = collections.deque()
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == 1:
                    q.append((i,j))
        day = 0            
        while q:
            num = len(q)
            day += 1  
            for k in range(num):
                x,y = q.popleft()
                for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
                    new_x = x + dx 
                    new_y = y + dy 
                    if 0<=new_x<len(grid) and 0<=new_y<len(grid[0]):
                        if grid[new_x][new_y] == 1 or grid[new_x][new_y] == 2 :
                            continue
                        elif grid[new_x][new_y] == 0:
                            grid[new_x][new_y] = 1 
                            q.append((new_x, new_y))
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == 0:
                    return -1 
        return day-1  #最后一天入队的新people只检查没感染
```

### Smallest Rectangle Enclosing Black Pixels

```python
class Solution:
    """
    @param image: a binary matrix with '0' and '1'
    @param x: the location of one of the black pixels
    @param y: the location of one of the black pixels
    @return: an integer
    """
    def minArea(self, image, x, y):
        # write your code here
        if not image or not image[0]:
            return 0 
        max_x = min_x = x    
        max_y = min_y = y 
        q = collections.deque([(x,y)])
        visited = set()
        while q:
            x,y = q.popleft()
            visited.add((x,y))
            for dx,dy in [(0,1), (0,-1),(1,0), (-1,0)]:
                newx = x + dx 
                newy = y + dy 
                if 0 <= newx < len(image) and 0 <= newy <len(image[0]):
                    if image[newx][newy] == '1' and (newx, newy) not in visited:
                        visited.add((newx, newy))
                        q.append((newx, newy))
                        if newx < min_x:
                            min_x = newx 
                        elif newx > max_x:
                            max_x = newx
                        if newy < min_y:
                            min_y = newy 
                        elif newy > max_y:
                            max_y = newy
        return (max_x - min_x + 1)*(max_y - min_y + 1)


#Version 2: binary search      
class Solution:
    """
    @param image: a binary matrix with '0' and '1'
    @param x: the location of one of the black pixels
    @param y: the location of one of the black pixels
    @return: an integer
    """
    def minArea(self, image, x, y):
        # Write your code here
        m = len(image)
        if m == 0:
            return 0
        n = len(image[0])
        if n == 0:
            return 0

        start = y
        end = n - 1
        while start < end:
            mid = start + (end - start) // 2 + 1
            if self.checkColumn(image, mid):
                start = mid
            else:
                end = mid - 1

        right = start

        start = 0
        end = y
        while start < end:
            mid = start + (end - start) // 2
            if self.checkColumn(image, mid):
                end = mid
            else:
                start = mid + 1

        left = start

        start = x
        end = m - 1
        while start < end:
            mid = start + (end - start) // 2 + 1
            if self.checkRow(image, mid):
                start = mid
            else:
                end = mid - 1

        down = start

        start = 0
        end = x
        while start < end:
            mid = start + (end - start) // 2
            if self.checkRow(image, mid):
                end = mid
            else:
                start = mid + 1

        up = start

        return (right - left + 1) * (down - up + 1)

    def checkColumn(self, image, col):
        for i in range(len(image)):
            if image[i][col] == '1':
                return True
        return False

    def checkRow(self, image, row):
        for j in range(len(image[0])):
            if image[row][j] == '1':
                return True
        return False
```

### Word Ladder

\(简单图最短路径\)

给出两个单词（_start_和_end_）和一个字典，找出从_start_到_end_的最短转换序列，输出最短序列的长度。

变换规则如下：

1. 每次只能改变一个字母。
2. 变换过程中的中间单词必须在字典中出现。\(起始单词和结束单词不需要出现在字典中\)

```python
class Solution:
    """
    @param: start: a string
    @param: end: a string
    @param: dict: a set of string
    @return: An integer
    """
    #分层遍历的BFS
    def ladderLength(self, start, end, dict):
        # write your code here
        dict.add(end)
        q = collections.deque([start])
        visited = set([start])
        distance = 0
        while q:
            distance += 1 
            for i in range(len(q)):
                word = q.popleft()
                if word == end:
                        return distance
                for next in self.next_words(word):
                    if next in dict and next not in visited:
                        q.append(next)
                        visited.add(next)
        return 0
        #ha sh ma p
    # O(26 * L^2)
    # L is the length of word
    def next_words(self, word):
        words = []
        for i in range(len(word)):
            left, right = word[:i], word[i+1:]#O(L)
            for letter in 'abcdefghijklmnopqrstuvwxyz':
                if word[i] == letter:
                    continue
                words.append(left+letter+right)
        return words         


#不使用分层遍历的版本。使用 distance 这个 hash 来存储距离来实现记录每个节点的距离。
    def ladderLength(self, start, end, dict):
        # write your code here
        dict.add(end)
        q = collections.deque([start])
        distance = {start: 1}
        while q:
            word = q.popleft()
            if word == end:
                    return distance[end]
            for next in self.next_words(word):
                if next in dict and next not in distance:
                    q.append(next)
                    distance[next] = distance[word] + 1
        return 0

    def next_words(self, word):
        words = []
        for i in range(len(word)):
            left, right = word[:i], word[i+1:]
            for letter in 'abcdefghijklmnopqrstuvwxyz':
                if word[i] == letter:
                    continue
                words.append(left+letter+right)
        return words
```

### Word Ladder II

给出两个单词（_start_和_end_）和一个字典，找出所有从_start_到_end_的最短转换序列。

变换规则如下：

1. 每次只能改变一个字母。
2. 变换过程中的中间单词必须在字典中出现。

```python
class Solution:
    """
    @param: start: a string
    @param: end: a string
    @param: dict: a set of string
    @return: a list of lists of string
    """
#v1: end-->start:bfs
#    start-->end:dfs
#从 end 到 start 做一次 BFS，并且把距离 end 的距离都保存在 distance 中。 然后在从 start 到 end 做一次 DFS，每走一步必须确保离 end 的 distance 越来越近。    
    def findLadders(self, start, end, dict):
        # write your code here
        paths = []
        dict.add(start)
        dict.add(end)
        distance = {end:1}

        self.bfs(dict,distance,end)
        self.dfs(dict,distance,paths,start,end,[start])
        #print (paths.append(['a','c']))
        return paths

    def bfs(self, dict, distance, end) :
        q = collections.deque([end])
        while q:
            word = q.popleft()
            for next in self.next_words(dict,word):
                if next not in distance:
                    distance[next] = distance[word] + 1 
                    q.append(next)

    def next_words(self, dict, word):
        words=[]
        for i in range(len(word)):
            left, right = word[:i], word[i+1:]
            for char in 'abcdefghijklmnopqrstuvwxyz':
                if char == word[i]:
                    continue
                if left+char+right in dict:
                    words.append(left+char+right)
        return words        

    def dfs(self, dict, distance, paths,cur,end, result):
        if cur == end:
            paths.append(list(result)) #results.append(path)中, path是一个reference，类似浅拷贝。list(path), 或者list[:]，相当于进行deep copy，然后再把copy加入results
            return 
        for next in self.next_words(dict, cur):
            if distance[next] == distance[cur] - 1 :
                result.append(next)
                self.dfs(dict,distance,paths,next ,end,result)
                result.pop()
```

### 总结 Conclusion

* 能用 BFS 的一定不要用 DFS\(除非面试官特别要求\)
* BFS 的两个使用条件

  • 图的遍历\(由点及面，层级遍历\)

  • 简单图最短路径

* 是否需要层级遍历

  • size = queue.size\(\)

* 拓扑排序必须掌握!
* 坐标变换数组

  • deltaX, deltaY • inBound

