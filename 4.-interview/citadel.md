# Citadel

## Initial Public Offering

### A. IPO \(Braze Version\)

  
 1. The bidder with the highest price gets the number of shares they bid for  
 2. If multiple bidders have bid at the same price, the bidders are assigned shares **in the order in which they placed their bids \(earliest bids first\)**

```python
#sort twice
#time:O(nologn)
#bids: id, num, price, time
import collections
def solution(bids, total):
    #init hash
    bids.sort(key = lambda x:(-x[2],x[3]),reverse = True)
    res = []
    while total>0:
        bid = bids.pop()
        print(bid)
        total -= bid[1]
        print(total)
    while bids:
        res.append(bids.pop()[0])
    return res


if __name__ == "__main__":
    #print(solution([[1, 2, 5, 0], [2, 1, 4, 2], [3, 5, 4, 6]],3))
    print(solution([[1, 5, 5, 0],[2, 7, 8, 1],[3, 7, 5, 1],[4, 10, 3, 3]],18))

```

###  B. IPO

  
1. The bidder with the highest price gets the number of shares they bid for  
2. If multiple bidders have bid at the same price, the bidders are assigned shares as follows:  
         **Each bidder in the same price group gets assigned once share each consecutively, with each bidder**   
         **being arranged inside the group based on their timestamps. Once a bidder gets the number of shares**   
         **they bid for, they will be removed from above iterative process and the process which then continues**   
         **until all bidders are removed or the shares get exhausted, whichever comes first.**

```python
#!/bin/python3

import math
import os
import random
import re
import sys



#
# Complete the 'getUnallottedUsers' function below.
#
# The function is expected to return an INTEGER_ARRAY.
# The function accepts following parameters:
#  1. 2D_INTEGER_ARRAY bids
#  2. INTEGER totalShares
#
import collections
def getUnallottedUsers(bids, totalShares):
    # Write your code here
    #init hash
    price = set()
    price_to_bids =  collections.defaultdict(list)
    for bid in bids:
        price.add(bid[2])
        price_to_bids[bid[2]].append(bid)
    #sort same_price_bids by time      
    for x in price_to_bids:
       price_to_bids[x].sort(key = lambda a:a[3])
    price = sorted(list(price))
    print(price)
    print(price_to_bids)
    #locate: 
    ans = []
    while totalShares>0 and price:
        cur_price = price.pop()
        cur_bids = price_to_bids[cur_price]
        if len(cur_bids) == 1:
            totalShares -= cur_bids[0][1]
        else:
            if len(cur_bids)<=totalShares:
                for x in cur_bids:
                    totalShares -= x[1]
            else:
                for _ in range(totalShares):
                    cur_bids.pop(0)
                for bid in cur_bids:
                    ans.append(bid[0])
                totalShares = 0
    #check and order
    while price:
        last_price = price.pop()
        for bid in price_to_bids[last_price]:
            ans.append(bid[0])
    ans.sort()
    return ans

if __name__ == '__main__':
```

## [Consecutive Sum](https://leetcode.com/problems/consecutive-numbers-sum/)

```python
#math
#time:O(sqr(N))
'''
x + (x+1) + (x+2)+...+ k terms = N
kx + k*(k-1)/2 = N ->
kx = N - k*(k-1)/2 ->
N-k*(k-1)/2>0
'''
class Solution:
    def consecutiveNumbersSum(self, N: int) -> int:
        k,count = 1,0
        while N > k*(k-1)//2:
            if (N - k*(k-1)/2)%k == 0:
                count += 1
            k += 1
        return count-1
```

## Matrix Summarization \(Before and After Matrix\)

```python
有一个 after matrix 是通过以下方式由 before matrix 得到的
要求返回 before matrix
s=0
for (i=0;  i<=x;  i++)
    for (j=0; j<=y; j++)
        s = s+before(i,j)
after(x,y)=s

从右往左每个 element 减去它左边的 element
从下往上每个 element 减去它上边的 element



#code
def solution(arr):
    rows,cols = len(arr), len(arr[0])
    before = [[0]*cols for i in range(rows)]
    #initialize
    before[0][0] = arr[0][0]
    for x in range(1,rows):
        before[x][0] = arr[x][0] - arr[x-1][0]
    for y in range(1, cols):
        before[0][y] = arr[0][y] - arr[0][y-1]
    for x in range(1,rows):
        for y in range(1,cols):
            before[x][y] = arr[x][y] - arr[x-1][y] - arr[x][y-1] + arr[x-1][y-1]
    return before
    
if __name__ == '__main__':
    print(solution([[2,5],[7,17]]))
```



