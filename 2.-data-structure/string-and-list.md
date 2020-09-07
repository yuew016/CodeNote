# 2.4 string & list

## [Reorder Data in Log Files](https://leetcode.com/problems/reorder-data-in-log-files/)

```python
class Solution(object):
    def reorderLogFiles(self, logs):
        """
        :type logs: List[str]
        :rtype: List[str]
        """
        letters = []
        digits = []
        for item in logs:
            if item.split()[-1].isdigit():
                digits.append(item)
            else:
                letters.append(item)
        letters.sort(key = lambda x: x.split(" ", 1)[::-1])
        return letters + digits
        
```

## Lexicographical Maximum Substring String

[https://www.geeksforgeeks.org/lexicographical-maximum-substring-string/](https://www.geeksforgeeks.org/lexicographical-maximum-substring-string/)

```python
# Python 3 program to find the 
# lexicographically maximum substring. 
def LexicographicalMaxString(str): 
	
	# loop to find the max leicographic 
	# substring in the substring array 
	mx = "" 
	for i in range(len(str)): 
		mx = max(mx, str[i:]) 

	return mx 

# Driver code 
if __name__ == '__main__': 
	str = "ababaa"
	print(LexicographicalMaxString(str)) 
	
# This code is contributed by 
# Sanjit_Prasad 

```

## 14. Longest Common Prefix

```python
#14:47
#scan the list and use point to record the max index matches with the previes one
class Solution:
    def longestCommonPrefix(self, m: List[str]) -> str:
        if not m: return ''
				#since list of string will be sorted and retrieved min max by alphebetic order
        m.sort()
        s1, s2 = m[0], m[-1]
        for i, c in enumerate(s1):
            if c != s2[i]:
                return s1[:i] #stop until hit the split index
        return s1
```

## 819. Most Common Word

```python
'''
c.isalnum(),isalpha(), isaldigit()
'''
import collections
class Solution:
    def mostCommonWord(self, paragraph: str, banned: List[str]) -> str:
        #remove ,.;!, keep alnum()
        p = "".join([c.lower() if c.isalpha() else " " for c in paragraph])
        p = p.split()
      #  print(p)
        p_dict = defaultdict(int)
        banned = set(banned)
        for w in p:
            if w not in banned:
                p_dict[w] += 1
      #  print(p_dict.items())
        return max(p_dict.items(), key = lambda x: x[1])[0]
        
            
        
        
        
```

## 3[8. Count and Say](https://leetcode.com/problems/count-and-say)

```python
'''
22:39
run n loops using two points to count.
time:O(nm), space:O(1)
'''
class Solution:
    def countAndSay(self, n: int) -> str:
        result = '1'
        for _ in range(n-1):
            prev = result
            result = ''
            l,r = 0,0
            while r<len(prev):
                while r<len(prev) and prev[l] == prev[r]:
                    r += 1
                result += str(r-l) + prev[l]
                l = r   
        return result

```

