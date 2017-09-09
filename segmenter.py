import math
from enum import Enum

import cv2
import numpy as np

from element import Element
from utils import convert_line_to_polar, quantize_color, get_common_color, join_elements


class Method(Enum):
    MORPHOLOGICAL_SEGMENTATION = 0
    COLOR_SEGMENTATION = 1


class Segmenter:
    def __init__(self):
        self.original_image = None

    def segment(self, image, method=Method.COLOR_SEGMENTATION):
        self.original_image = image
        # Find the splitting lines
        lines = self.__find_lines(image)
        if lines:
            # Split the image if there are lines
            elements = self.__split_image(image, lines)
        else:
            # or take it as a single element

            elements = [Element(image, (0, 0))]
        if method == Method.COLOR_SEGMENTATION:
            segmentation_function = self.__color_segment
        elif method == Method.MORPHOLOGICAL_SEGMENTATION:
            segmentation_function = self.__morpholgical_segment
        else:
            raise ValueError("Invalid argument for method")

        temp = []
        for element in elements:
            temp.extend(segmentation_function(element))
        elements = temp
        del temp

        for element in elements:
            cv2.rectangle(self.original_image, (element.x, element.y),
                          (element.x + element.width, element.y + element.height), (255, 0, 255))

        cv2.imshow(method.name, self.original_image)
        cv2.waitKey(0)
        return elements

    def __find_lines(self, image):
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edged = cv2.Canny(grayImage, 180, 200)
        lines = []
        for i in range(2):
            if i == 0:  # Vertical lines
                threshold = image.shape[1] // 5
                filter_fn = lambda theta: 0.505 * math.pi >= abs(theta) >= 0.495 * math.pi
                dilate_se = np.zeros((3, 9), dtype=np.uint8)
                dilate_se[..., 4] = 1
            else:
                threshold = image.shape[0] // 5
                filter_fn = lambda theta: abs(theta) < 0.05
                dilate_se = np.zeros((9, 3), dtype=np.uint8)
                dilate_se[4, ...] = 1
            dilated = cv2.dilate(edged, dilate_se)
            hp_lines = cv2.HoughLinesP(dilated, 1, math.pi / 180, threshold=threshold, minLineLength=threshold)

            polar_lines = []
            # Convert detected lines to polar form
            if hp_lines is not None:
                for l in hp_lines:
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
                    if not filter_fn(theta):
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
                        line_group.clear()
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
                lines.extend(temp)
        return lines

    def __split_image(self, image, lines):
        """
        Splits the images according to the vertical and horizontal split lines
        Assumption: Text is in horizontal or vertical orientation
        """
        elements = []
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

        # Split the image into vertical elements
        first_x = 0
        vertical_elements = []
        for l in vertical_lines:
            max_x = max(l[0], l[2])
            if (max_x - first_x) < image.shape[1] // 20:
                continue
            element_image = image[:, first_x:max_x]
            element = Element(element_image, (first_x, 0))
            vertical_elements.append(element)
            first_x = max_x

        element_image = image[:, first_x:]
        element = Element(element_image, (first_x, 0))
        vertical_elements.append(element)

        # Split the vertical elementss using the horizontal lines
        for v_element in vertical_elements:
            first_y = 0
            for l in horizontal_lines:
                max_y = max(l[1], l[3])
                if (max_y - first_y) < image.shape[1] // 20:
                    continue
                element_image = v_element.image[first_y:max_y, :]
                if element_image.size != 0:
                    element = Element(element_image, (v_element.location[0], first_y))
                    elements.append(element)
                first_y = max_y

            element_image = v_element.image[first_y:, :]
            element = Element(element_image, (v_element.location[0], first_y))
            if element.size != 0:
                elements.append(element)

        return elements

        return portions
    def __color_segment(self, element):
        """

        :rtype: list
        :param element: Element
        :return: elements
        """
        image = element.image
        temp = image.copy()

        temp = quantize_color(temp)
        color = get_common_color(temp)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                pixel = temp[i, j, :]
                match = [x for x, y in zip(pixel, color) if x == y]
                if len(match) == 3:
                    temp[i, j, :] = 0
                else:
                    temp[i, j, :] = 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        temp = cv2.dilate(temp, kernel)
        temp = temp[:, :, 0]

        num_labels, labels, stats = cv2.connectedComponentsWithStats(temp)[0:3]
        elements = []
        for i in range(num_labels):
            stat = stats[i]
            top = stat[cv2.CC_STAT_TOP]
            left = stat[cv2.CC_STAT_LEFT]
            width = stat[cv2.CC_STAT_WIDTH]
            height = stat[cv2.CC_STAT_HEIGHT]
            element_image = image[top:top + height, left:left + width, :]
            # If there is enclosing box it will be detected as connected shape so if there is portion with size of
            # 90% or more of the original size, neglect it
            if (height * width) / (element.shape[0] * element.shape[1]) > 0.9:
                continue
            else:
                e = Element(element_image, (element.x + left, element.y + top))
                elements.append(e)

        elements = self.__filter_elements(elements, element)
        return join_elements(elements, element)
        img = portion.image
    def __morpholgical_segment(self, element):
        img = element.image
        # 1. Up-sample the image
        img = cv2.pyrUp(img)

        # 2. Find Gaussian Blur
        img = cv2.GaussianBlur(img, (5, 5), 5)

        # 3. Find OTSU threshold & apply Canny
        th, bw1 = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)

        bw1 = cv2.Canny(img, th, th / 2, apertureSize=5, L2gradient=True)
        bw2 = cv2.Canny(img, th, th / 2, apertureSize=3, L2gradient=True)  # less noisy version
        # 4. Apply Hit or miss morpholgical operation to connect broken lines
        kernels = [
            np.array([[0, 0, 1], [0, 0, 0], [0, 0, 0]]),
            # np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]])

        ]
        for kernel in kernels:
            hitmiss = cv2.morphologyEx(bw1, cv2.MORPH_HITMISS, kernel)

            bw1 = cv2.bitwise_or(hitmiss, bw1)

        # 5. apply morphological reconstruction to less noisy image using the modified noisy image */
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        prevMu = 0.0
        recons = np.copy(bw2)
        for i in range(200):

            recons = cv2.dilate(recons, kernel)
            recons &= bw1
            mu = np.mean(recons)
            if abs(mu - prevMu) < 0.001:
                break

            prevMu = mu

        num_labels, labels, stats = cv2.connectedComponentsWithStats(recons)[0:3]
        elements = []
        for i in range(num_labels):
            stat = stats[i]
            top = stat[cv2.CC_STAT_TOP] // 2
            left = stat[cv2.CC_STAT_LEFT] // 2
            width = stat[cv2.CC_STAT_WIDTH] // 2
            height = stat[cv2.CC_STAT_HEIGHT] // 2
            subelement_image = img[top:top + height, left:left + width, :]
            # If there is enclosing box it will be detected as connected shape so if there is portion with size of
            # 90% or more of the original size, neglect it
            if (height * width) / (element.shape[0] * element.shape[1]) > 0.9:
                continue
            else:
                e = Element(subelement_image, (element.x + left, element.y + top))
                elements.append(e)

        elements = self.__filter_elements(elements, element)

        return join_elements(elements, element)

    @staticmethod
    def __filter_elements(elements, parent_element):
        avg_width = np.mean([e.width for e in elements])
        avg_height = np.mean([e.height for e in elements])
        if parent_element.width / parent_element.height > 2:
            w_tolerence = parent_element.width // 5
            h_tolerence = parent_element.height
        elif parent_element.height / parent_element.width > 2:
            w_tolerence = parent_element.width
            h_tolerence = parent_element.height // 5
        elif 1.05 >= parent_element.height / parent_element.width >= 0.95:
            w_tolerence = parent_element.width // 5
            h_tolerence = parent_element.height // 5
        else:
            w_tolerence = parent_element.width
            h_tolerence = parent_element.height
        elements = filter(lambda
                              e: avg_width + w_tolerence >= e.width >= avg_width - w_tolerence and avg_height + h_tolerence >= e.height >= avg_height - h_tolerence,
                          elements)
        elements = list(elements)
        return elements
