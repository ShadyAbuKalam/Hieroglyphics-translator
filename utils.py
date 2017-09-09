import math
import operator
import random

import cv2
import numpy as np

from element import Element


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
            theta = math.pi / 2
        else:
            theta = 1.5 * math.pi

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


def quantize_color(image, k=2):
    Z = image.reshape((-1, 3))

    # convert to np.float32
    Z = np.float32(Z)

    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv2.kmeans(Z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    return res.reshape(image.shape)


def get_common_color(image, quantize=False):
    if quantize:
        image = quantize_color(image)
    hist = {}
    for i in range(100):
        pixel = image[random.randint(0, image.shape[0] - 1), random.randint(0, image.shape[1] - 1), :].tolist()
        pixel = [str(i) for i in pixel]
        pixel = ':'.join(pixel)

        hist[pixel] = hist.get(pixel, 0) + 1

    k = max(hist.items(), key=operator.itemgetter(1))[0]
    color = k.split(':')
    color = [int(i) for i in color]
    return color


def intersect(e1, e2):
    """
    Checks if two picture elements intersect

    :param e1: Element
    :param e2: Element
    :return: True for intersection, otherwise no
    :rtype: bool
    """
    no_intersect = e1.x + e1.width < e2.x or e2.x + e2.width < e1.x or e1.y + e1.height < e2.y or e2.y + e2.height < e1.y
    return not no_intersect


def join_elements(elements, original_portion) :
    """

    :rtype: list of Element
    :param elements:
    :param original_portion: Element
    :return: A list of elements where all intersected elements are joined together
    """
    while True:
        changed = False

        for i in range(len(elements)):
            if elements[i] is None:
                continue
            else:
                e1 = elements[i]
            for j in range(i + 1, len(elements)):
                if elements[j] is None:
                    continue
                else:
                    e2 = elements[j]

                if intersect(e1, e2):
                    x_points = [e1.x, e1.x + e1.width, e2.x, e2.x + e2.width]
                    y_points = [e1.y, e1.y + e1.height, e2.y, e2.y + e2.height]
                    start_x = min(x_points)
                    width = max(x_points) - start_x

                    start_y = min(y_points)
                    height = max(y_points) - start_y
                    subportion = original_portion.image[
                                 start_y - original_portion.y:start_y - original_portion.y + height,
                                 start_x - original_portion.x:start_x + width - original_portion.x, :]

                    elements[i] = Element(subportion, (start_x, start_y))
                    elements[j] = None
                    changed = True
        if not changed:
            break
    return list(filter(lambda e:e is not None,elements))



if __name__ == "__main__":
    convert_line_to_polar(-3, 0, 0, -3)
