class Counter:
    def __init__(self):
        self.total = 0

    def add(self, v):
        self.total = self.total + v


def run(c, n):
    for i in range(n):
        c.add(i)
    return c.total


def main():
    print(run(Counter(), @N@))


main()
