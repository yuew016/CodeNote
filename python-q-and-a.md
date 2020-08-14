# 0. python Q&A

## 0.1 list, set, dict, tuple

## 0.2 range范围

```text
range(stop)
range(start, stop）
range(start, stop, step)
```

参数说明：

* start: 计数从 start 开始。默认是从 0 开始。例如range（5）等价于range（0， 5）;
* stop: 计数到 stop 结束，但不包括 stop。例如：range（0， 5） 是\[0, 1, 2, 3, 4\]没有5
* step：步长，默认为1。例如：range（0， 5） 等价于 range\(0, 5, 1\)

## 0.3 list、stack、queue、linklist、dict

### 1\) list

python中的列表可以**混合存储任意数据类型**，因为列表中存的是元素的**内存地址**，而不是元素的值，

比如有列表li = \[1, True, 'hello', {'k': 'v'}\]。

li中的元素在内存中**并不是连续存储**的，因此li存储这些元素所在的内存地址，对元素取值时，先在列表中找到元素的内存地址，再通过内存地址找到元素

![img](https://img-blog.csdn.net/20171114063740628?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvQXloYW5faHVhbmc=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

列表没有长度限制是如何实现的呢？当我们通过append方法向列表中添加元素时，如果列表满了，那么新申请2倍于原来列表的内存地址，并将原列表的值拷贝到新的内存地址中。

list:链表,有序, 通过索引进行查找,使用方括号”\[\]”;

倒序访问：A\[-1\]

增：append\(\): 添加到尾部，insert\(\)方法添加到指定位置（下标从0开始）

​ extend\(\)

​ +

​ list1=\[1, 2, 3\] .list2=\[4, 5, 6\]

list1.append\(list2\) = \[1, 2, 3, \[4, 5, 6\]\]

list1.extend\(list2\) = \[1, 2, 3, 4, 5, 6\]

List1 + list2 = \[1, 2, 3, 4, 5, 6\]

删：pop\(\)删除最后尾部元素，也可以指定一参数删除指定位置

### 2\) stack

后进先出（last-in, first-out）

栈的基本操作:

定义： stack = \[\]

1. 进栈：append
2. 出栈：pop\(\)
3. 查看栈顶：stack\[-1\] 直接取索引

### 3\) deque

Python的deque模块可以实现队列，并且支持双向队列：

```python
from collections import deque

li = [1,2,3,4]
queue = deque(li) # 从列表创建双向队列（也可以直接创建）
queue.append(5)  # deque([1, 2, 3, 4, 5])
queue.popleft()  # deque([2, 3, 4, 5])

#双向队列对首进队，队尾出对

queue.appendleft('a')  # deque(['a', 2, 3, 4, 5])
queue.pop()  # deque(['a', 2, 3, 4])
```

### 4\) 链表

区别与数组/列表：数组和列表是连续存储的，删除和插入元素，其它元素需要补过去或者后移，时间复杂度是O\(n\)；

而链表则不会这样，它的时间复杂度是O\(1\)

```python
#定义节点和构造链表

#定义节点类

class Node:
    def __init__(self, val):
        self.val = val
        self.next = None

#实例化节点

a = Node(1)
b = Node(2)
c = Node(3)

#构造链表
a.next = b
b.next = c

#print head
head = a
print(head)#print a, b, c 

a = b
print(head)#print a, b, c ,(soft copy)
```

### 5\) dict

字典,字典是一组键\(key\)和值\(value\)的组合,通过键\(key\)进行查找,没有顺序, 使用大括号”{}”;

### 6\) set

:集合,无序,元素只出现一次, 自动去重,使用”set\(\[\]\)”

add\(x\)和remove\(x\)

## 0.4 str,list,set间的转换

```python
a = '123abbcc!@#'  
    str -> list: list(a)
    str -> set: set(a)

b = ['1', '2', '3', 'a', 'b', 'c', '!', '@', '#']
    list -> str: "".join(b)
  list -> set: set(b)

c =  set(['a', '!', 'c', 'b', '@', '#', '1', '3', '2']) 
    set->str: "".join(c)
  set->list: list(c)
```

## 0.5 setdefault\(\) and get\(\)

`dict.setdefault(key, default=None)`–&gt; 有key获取值，否则设置 key:default，并返回default，default默认值为None

`dict.get(key, default=None)`–&gt; 有key获取值，否则返回default。default默认值为None。

## 0.6 string split\(\)



```text
str.split(str="", num=string.count(str)).
```

#### 参数

* str -- 分隔符，默认为所有的空字符，包括空格、换行\(\n\)、制表符\(\t\)等。
* num -- 分割次数。默认为 -1, 即分隔所有

## 0.7 input, output

```python
if __name__ == '__main__':
    T = int(input())

    for _ in range(T):
        n=int(input())
        arr = [int(x) for x in input().split()]

        print(maxSum(arr,n))
```

