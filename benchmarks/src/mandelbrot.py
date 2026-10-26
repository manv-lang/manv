def mandel(cr, ci, limit):
    zr = 0.0
    zi = 0.0
    i = 0
    while i < limit:
        zr2 = zr * zr
        zi2 = zi * zi
        if zr2 + zi2 > 4.0:
            return i
        zi = 2.0 * zr * zi + ci
        zr = zr2 - zi2 + cr
        i = i + 1
    return limit


def main():
    total = 0
    for y in range(@N@):
        for x in range(@N@):
            cr = 3.0 * float(x) / float(@N@) - 2.0
            ci = 2.0 * float(y) / float(@N@) - 1.0
            total = total + mandel(cr, ci, @M@)
    print(total)


main()
