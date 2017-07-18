import math


def convert_line_to_polar(x1, y1, x2, y2):
    if x1 == x2:
        if x1 >= 0:
            theta = 0
        else:
            theta = math.pi
        rho = x1
    elif y1 == y2:
        rho = y1
        if y1 >= 0:
            theta = math.pi/2
        else:
            theta = 1.5*math.pi

    else:
        theta = math.atan2(y1 - y2, x1 - x2) - math.pi / 2
        m = (y1 - y2) / (x1 - x2)
        c = (y1 + y2 - m * (x1 + x2)) / 2

        y0 = c
        x0 = -c / m

        rho = x0 * math.cos(theta) + y0 * math.sin(theta)

        if rho < 0:
            rho = abs(rho)
            theta += math.pi
    while theta < 0:
        theta += 2 * math.pi
    while theta > 2 * math.pi:
        theta -= 2 * math.pi
    return rho, theta


if __name__ == "__main__":
    convert_line_to_polar(-3, 0, 0, -3)
