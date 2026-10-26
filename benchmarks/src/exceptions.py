def main():
    caught = 0
    for i in range(@N@):
        try:
            raise ValueError("x")
        except ValueError:
            caught = caught + 1
    print(caught)


main()
