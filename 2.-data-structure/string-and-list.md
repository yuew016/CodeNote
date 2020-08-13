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

