formatter = {}


class TableMeta(type):
    def __init__(cls, clsname, bases, methods):
        super().__init__(clsname, bases, methods)
        if hasattr(cls, 'name'):
            formatter[cls.name] = cls


class A(metaclass=TableMeta):
    name = 'A'


class B(A):
    name = 'B'

    def __init__(self):
        pass


# b = A()
print(formatter)
