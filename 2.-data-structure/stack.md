# 2.1 Stack

## Stack

利用栈暂且保存有效信息 

翻转栈的运用

栈优化dfs，变成非递归

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

## 316 Remove Duplicate Letters

```python
 ## RC ##
		## APPROACH : STACK ##
		## 100% Same Problem Leetcode 1081. Smallest Subsequence of Distinct Characters ##
		## LOGIC ##
		#	1. We add element to the stack
		#	2. IF we get bigger element, we just push on top
		#	3. ELSE we pop if and only if there are other occurances of same letter again in the string, otherwise we donot pop
		#	4. If an element is already in the stack, we donot push.
		## TIME COMPLEXICITY : O(N) ##
		## SPACE COMPLEXICITY : O(N) ##
class Solution:
    def removeDuplicateLetters(self, s: str) -> str:
        stack = []
        last_position = {c:i for i,c in enumerate(s)}
        for i,c in enumerate(s):
            if c in stack:
                continue
            while stack and stack[-1]>c and i<last_position[stack[-1]]:
                stack.pop()
            stack.append(c)
          # print(stack)
        return "".join(stack)
        
```

