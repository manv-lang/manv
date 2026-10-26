def make_adder(n):
    def add(x):
        return x + n
    return add


def run(f, n):
    acc = 0
    for i in range(n):
        acc = f(acc)
    return acc


def main():
    print(run(make_adder(3), @N@))


main()
