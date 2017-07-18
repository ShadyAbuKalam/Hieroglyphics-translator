import cv2
import math
import numpy as np
import sys

from utils import convert_line_to_polar


class Segmenter:
    def __init__(self):
        self.original_image = None

    def segment(self, image):
        self.original_image = image
        lines = self.__find_lines(image)
        if lines:
            portions = self.__split_image(image, lines)
        else:
            portions = [image]

    def __find_lines(self, image):
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edged = cv2.Canny(grayImage, 180, 200)
        threshold = min(image.shape[0:2]) // 5
        lines = cv2.HoughLinesP(edged, 1, math.pi / 180, threshold=threshold, minLineLength=threshold)

        polar_lines = []
        # Convert detected lines to polar form
        if lines is not None:
            for l in lines:
                x1 = l[0][0]
                y1 = l[0][1]
                x2 = l[0][2]
                y2 = l[0][3]
                l = convert_line_to_polar(x1, y1, x2, y2)
                polar_lines.append(l)

        # Filter any non-vertical or non-horizontal line
        if polar_lines is not None:
            polar_lines = sorted(polar_lines, key=lambda x: x[0])
            temp = []
            for l in polar_lines:
                theta = l[1]
                theta = np.unwrap([theta])[0]
                if abs(theta) > 0.05 and (abs(theta) > 0.505 * math.pi or abs(theta) < 0.495 * math.pi):
                    continue
                else:
                    temp.append(l)

            polar_lines = temp.copy()
            temp.clear()

            # Group nearby lines together and pick the median
            line_group = set()
            for i in range(len(polar_lines)):
                if i == len(polar_lines) - 1:
                    if len(line_group) != 0:
                        rho = np.median([l[0] for l in line_group])
                        l = list(line_group.pop())
                        l[0] = rho
                        temp.append(tuple(l))
                        if rho - polar_lines[i][0] > min(image.shape[0:2]) // 40:
                            temp.append(tuple(polar_lines[i]))

                    else:
                        temp.append(tuple(polar_lines[i]))
                    continue

                rho = polar_lines[i][0]
                theta = polar_lines[i][1]
                n_rho = polar_lines[i + 1][0]
                n_theta = polar_lines[i + 1][1]
                if abs(rho - n_rho) < min(image.shape[0:2]) // 20 and abs(math.cos(n_theta - theta)) > 0.99:
                    line_group.add(tuple(polar_lines[i]))
                    line_group.add(tuple(polar_lines[i + 1]))
                elif len(line_group) == 0:
                    temp.append(tuple(polar_lines[i]))
                else:
                    rho = np.median([l[0] for l in line_group])
                    l = list(line_group.pop())
                    l[0] = rho
                    temp.append(tuple(l))
            polar_lines = temp.copy()
            temp.clear()

            # Convert back polar_lines to 4-points notation
            for l in polar_lines:
                rho = l[0]
                theta = l[1]
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                temp.append((x1, y1, x2, y2))

            return temp

    def __split_image(self, image, lines):
        """
        Splits the images according to the vertical and horizontal split lines
        Assumption: Text is in horizontal or vertical orientation
        """
        portions = []
        horizontal_lines = []
        vertical_lines = []
        # Split lines into horizontal or vertical
        for l in lines:
            if l[2] - l[0] == 0:
                vertical_lines.append(l)
                continue
            theta = math.atan((l[3] - l[1]) / (l[2] - l[0]))
            if (abs(theta) < math.pi / 8):
                horizontal_lines.append(l)
            else:
                vertical_lines.append(l)

        # Split the image into vertical portions
        first_x = 0
        vertical_portions = []
        for l in vertical_lines:
            max_x = max(l[0], l[2])
            if (max_x - first_x) < image.shape[1] // 20:
                continue
            portion = image[:, first_x:max_x]
            vertical_portions.append(portion)
            first_x = max_x

        portion = image[:, first_x:]
        vertical_portions.append(portion)

        # Split the vertical portions using the horizontal lines
        for v_portion in vertical_portions:
            first_y = 0
            for l in horizontal_lines:
                max_y = max(l[1], l[3])
                if (max_y - first_y) < image.shape[1] // 20:
                    continue
                portion = v_portion[first_y:max_y, :]
                portions.append(portion)
                first_y = max_y
            portion = v_portion[first_y:, :]
            portions.append(portion)
        for portion in portions:
            cv2.imshow("Trial", portion)
            cv2.waitKey()
        return portions
