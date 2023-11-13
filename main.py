import argparse
import cv2
import numpy as np
from PIL.Image import Image 
import matplotlib.pyplot as plt

class BarCodeDector:
    """
    Detects bar codes in the given image, draws a border around them. 

    Algorithm outline:
        1. sharr gradient in x and y direction
        2. subtract y from x gradient for finding barcode region
        3. blur and threshold
        4. Closing kernel
        5. Series of dilations and erosions
        6. Largest contour should be the barcode
    """
    image_path: str
    """
    Path to the image possibly containing a barcode
    """
    border_color: tuple[int, int, int] 
    """
    Color to apply to the border drawn around the detected barcode, format: rgb
    """

    def __init__(self, image_path: str, border_color: str):
        if not image_path:
            raise ValueError("Expected 'image_path' to be a value, not None")
        self.image_path = image_path
        self.border_color = self.string_rgb_to_color(border_color)

    def string_rgb_to_color(self, rgb_fmt: str) -> tuple[int, int, int]:
        c = rgb_fmt.split(":")

        if len(c) != 3:
            raise ValueError("Expected 'border_color' format to be 'r:g:b', each value between 0 and 255")

        try:
            c = [int(x) for x in c]
        except:
            raise ValueError("Expected color channel in 'border-color' to be parsable as a number")

        # ugly, but keeps my type checker silent, i would prefer 'tuple(c)'
        return (c[0], c[1], c[2])

    def show_image(self, image):
        plt.imshow(image)
        plt.show()

    def sharr_gradient(self, image: Image) -> Image:
        depth = cv2.CV_32F
        x_gradient = cv2.Sobel(image, ddepth=depth, dx=1, dy=0, ksize=-1)
        y_gradient = cv2.Sobel(image, ddepth=depth, dx=0, dy=1, ksize=-1)
        result = cv2.subtract(x_gradient, y_gradient)
        return cv2.convertScaleAbs(result)

    def filter_noise(self, image: Image) -> Image:
        blur = cv2.blur(image, (9,9))
        (_, thresh) = cv2.threshold(blur, 225, 255, cv2.THRESH_BINARY)
        return thresh

    def closing_kernel(self, image: Image) -> Image:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
        return cv2.morphologyEx(image, cv2.MORPH_CLOSE, k)

    def eroding_dilation(self, image: Image) -> Image:
        image = cv2.erode(image, None, iterations = 4)
        return cv2.dilate(image, None, iterations = 4)

    def contours(self, image: Image) -> np.intp:
        (cnts, _) = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]

        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        return np.intp(box)

    def draw_box(self, image: Image, box: np.intp) -> Image:
        return cv2.drawContours(image, [box], -1, self.border_color, 3)

    def process(self):
        """
        Main function for processing the image
        """
        og_img = cv2.imread(self.image_path)
        # original image
        # self.show_image(image)

        # image is already greyscale, no need to convert (or so i thought got a cv2 error format mismatch, ig i have to greyscale this)
        image = cv2.cvtColor(og_img, cv2.COLOR_BGR2GRAY)

        image = self.sharr_gradient(image)
        # image with sharr gradient
        # self.show_image(image)

        image = self.filter_noise(image)
        # image with filtered noise
        # self.show_image(image)

        image = self.closing_kernel(image)
        # image with closing kernel
        # self.show_image(image)

        image = self.eroding_dilation(image)
        # eroded & dilated image
        # self.show_image(image)

        contours = self.contours(image)
        image = self.draw_box(og_img, contours)

        self.show_image(image)

if __name__ == "__main__":
    flag = argparse.ArgumentParser()
    flag.add_argument("-p", "--path", help="Path to the input image", required=True)
    flag.add_argument("-c", "--color", help="Color of the border, fmt: r:g:b", required=False, default="0:255:0")
    args = flag.parse_args()
    bc = BarCodeDector(args.path, args.color)
    bc.process()
else:
    raise RuntimeError("Script intended to be ran, not included")
