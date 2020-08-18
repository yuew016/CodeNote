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

## Maximum Substring

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

