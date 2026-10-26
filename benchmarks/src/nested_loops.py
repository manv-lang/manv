def main():
    total = 0
    for i in range(@N@):
        for j in range(@N@):
            total = total + (i * j) % 7
    print(total)


main()
