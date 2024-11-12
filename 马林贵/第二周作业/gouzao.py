class Father(object):
    def __init__(self, name):
        self.name = name
        print("name: %s" % (self.name))

    def getName(self):
        return 'Father ' + self.name


class Son(Father):
    def __init__(self, name):
        super(Son, self).__init__(name)  # 两种形式都可以
        # Father.__init__(self, name)
        print("hi")
        self.name = name

    def getName(self):
        return 'Son ' + self.name


if __name__ == '__main__':
    son = Son('runoob')
    print(son.getName())
