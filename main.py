import pathlib

import cv2

from detector import  Detector
from segmenter import Segmenter

if __name__ == "__main__":
    segmenter = Segmenter()
    for f in pathlib.Path("samples").iterdir():
            print("Processing {0}".format(f))
            image = cv2.imread(f.as_posix())
            letters = segmenter.segment(image)