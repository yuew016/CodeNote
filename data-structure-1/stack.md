# Stack

## Stack

### [394 Decode String](https://leetcode.com/problems/decode-string/)

```python
#time: O(n) space:O(n)
class Solution:
    def decodeString(self, s: str) -> str:
        if not s:
            return s
        
        num_stack = []
        str_stack = []
        res = ''
        num = 0
        for i in range(len(s)):
            if s[i].isdigit():
                num = num*10 + int(s[i])
            elif s[i] == '[':
                num_stack.append(num)
                num = 0
                str_stack.append(res)
                res = ''
            elif s[i] == ']':
                pre = str_stack.pop()
                res = pre + res*num_stack.pop()
            else:
                res += s[i]
        return res
```

