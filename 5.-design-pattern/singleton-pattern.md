# Singleton Pattern

## 使用函数装饰器实现单例

使用不可变的**类地址**作为键，其实例作为值，每次创造实例时，首先查看该类是否存在实例，存在的话直接返回该实例即可，否则新建一个实例并存放在字典中。

```python
def singleton(cls):
    _instance = {}
    def inner():
        if cls not in _instance:
            _instance[cls] = cls()
        return _instance[cls]
    return inner
    
@singleton
class A(object):
    def __init__(self):
        pass

#test
cls1 = A()
cls2 = A()
print(id(cls1) == id(cls2))
```

## 使用类装饰器实现单例

```python
class Singleton(object):
    def __init__(self, cls):
        self._cls = cls
        self._instance = {}
    def __call__(self):
        if self._cls not in self._instance:
            self._instance[self._cls] = self._cls()
        return self._instance[self._cls]

@Singleton
class Cls2(object):
    def __init__(self):
        pass

cls1 = Cls2()
cls2 = Cls2()
print(id(cls1) == id(cls2))


#同时，由于是面对对象的，这里还可以这么用
class Cls3():
    pass
Cls3 = Singleton(Cls3)
cls3 = Cls3()
cls4 = Cls3()
print(id(cls3) == id(cls4))
```

## 使用 \_\_new\_\_ 关键字实现单例

```python
class Single(object):
    _instance = None
    def __new__(cls, *args, **kw):
        if cls._instance is None:
            cls._instance = object.__new__(cls, *args, **kw)
        return cls._instance
    def __init__(self):
        pass

single1 = Single()
single2 = Single()
print(id(single1) == id(single2))
```

在理解到 \_\_new\_\_ 的应用后，理解单例就不难了，这里使用了

```text
_instance = None
```

来存放实例，如果 \_instance 为 None，则新建实例，否则直接返回 \_instance 存放的实例。

## 使用 metaclass 实现单例



同样，我们在类的创建时进行干预，从而达到实现单例的目的。

在实现单例之前，需要了解使用 type 创造类的方法，代码如下：

```text
def func(self):
    print("do sth")

Klass = type("Klass", (), {"func": func})

c = Klass()
c.func()
```

以上，我们使用 type 创造了一个类出来。这里的知识是 mataclass 实现单例的基础。

```text
class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class Cls4(metaclass=Singleton):
    pass

cls1 = Cls4()
cls2 = Cls4()
print(id(cls1) == id(cls2))
```

这里，我们将 metaclass 指向 Singleton 类，让 Singleton 中的 type 来创造新的 Cls4 实例。  


