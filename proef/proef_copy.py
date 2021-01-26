import copy

class Test1():
    def __init__(self):
        self.data = {"aap": {"noot": 1}}
        self.data2 = [1,2,3]


class Test2():
    def __init__(self, test1):
        self.data = copy.deepcopy(test1.data)
        self.data2 = copy.deepcopy(test1.data2)


t1 = Test1()
t2 = Test2(t1)
t2.data["aap"]["noot"] = 5
t2.data2.pop()
print(t1.data)
print(t1.data2)