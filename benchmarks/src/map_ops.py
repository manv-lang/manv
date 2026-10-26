def main():
    m = {}
    for i in range(@N@):
        m[str(i)] = i
    total = 0
    for i in range(@N@):
        total = total + m[str(i)]
    print(total)


main()
