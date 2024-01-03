# Author 9525469 
# Date 2023-11-10T16:16:00
"""
BarCodeDector

    usage: main.py [-h] -p PATH [-c COLOR]

    options:
      -h, --help                show this help message and exit
      -p PATH, --path PATH      Path to the input image
      -c COLOR, --color COLOR   Color of the border, fmt: r:g:b

Example usage:

    python3 barcodescanner.py -p barcode-1.tif
    python3 barcodescanner.py --path barcode-0.tif --color "255:0:0"
    python3 barcodescanner.py -p barcode-0.tif -c "255:0:0"

Can not be included as a library, will throw an RuntimeError.
"""
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
    Color to apply to the border drawn around the detected barcode, format: (r,g,b), defaults to (0,255,0)
    """

    def __init__(self, image_path: str, border_color: str):
        """
        Detects bar codes in the given image, draws a border around them. 

        Algorithm outline:
            1. sharr gradient in x and y direction
            2. subtract y from x gradient for finding barcode region
            3. blur and threshold
            4. Closing kernel
            5. Series of dilations and erosions
            6. Largest contour should be the barcode

        @args image_path path to the input image
        @args border_color rgb color in format r:g:b
        @error ValueError if image_path is empty
        """
        # image path is required
        if not image_path:
            raise ValueError("Expected 'image_path' to be a value, not None")
        self.image_path = image_path
        self.border_color = self.string_rgb_to_color(border_color)

    def string_rgb_to_color(self, rgb_fmt: str) -> tuple[int, int, int]:
        """
        converts rgb_fmt of format 'r:g:b' to a tuple [r, g, b] all of type
        int. 

        @arg rgb_fmt input format
        @error ValueError if invalid amount of arguments in format or number couldn't be parsed
        """
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
        """
        Displays the image via matplotlib

        @arg image image to display
        """
        plt.imshow(image)
        plt.show()

    def sharr_gradient(self, image: Image) -> Image:
        """
        Applies the sharr gradient[1] to the given image, applies
        convertScaleAbs[2] to the result and returns it, this highlights the
        barcode region.

        [1]: https://en.wikipedia.org/wiki/Sobel_operator
        [2]: https://docs.opencv.org/3.4/d2/de8/group__core__array.html#ga3460e9c9f37b563ab9dd550c4d8c4e7d

        @arg image image to process
        """
        x_gradient = cv2.Sobel(image, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
        y_gradient = cv2.Sobel(image, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)
        result = cv2.subtract(x_gradient, y_gradient)

        # converts to 8bit img while scaling and computing absolut values
        return cv2.convertScaleAbs(result)

    def filter_noise(self, image: Image) -> Image:
        """
        Attempts to filter possibly obstructing noise out of the given image by
        blurring with the gaussian method[1] and applying a treshold[2].

        [1]: https://docs.opencv.org/3.4/d4/d86/group__imgproc__filter.html#gaabe8c836e97159a9193fb0b11ac52cf1
        [2]: https://docs.opencv.org/3.4/d7/d1b/group__imgproc__misc.html#gae8a4a146d1ca78c626a53577199e9c57

        @arg image image to process
        """
        # (9,9) worked the best for filtering hard edges
        blur = cv2.GaussianBlur(image, (9,9), 0)

        # treshold found by testing to be 237
        (_, thresh) = cv2.threshold(blur, 237, 255, cv2.THRESH_BINARY)
        return thresh

    def closing_kernel(self, image: Image) -> Image:
        """
        Applies a closing operation[1] with a cv2.MORPH_RECT kernel[2] to the image

        [1]: https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#ga7be549266bad7b2e6a04db49827f9f32
        [2]: https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#gac2db39b56866583a95a5680313c314ad

        @arg image image to process
        """
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (29, 29))
        return cv2.morphologyEx(image, cv2.MORPH_CLOSE, k)

    def eroding_dilation(self, image: Image) -> Image:
        """
        Used for removing boundaries of foreground objects. Eroding removes
        white noise[1] while shrinking the object, therefore we dilate[2] the object.

        [1]: https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#gaeb1e0c1033e3f6b891a25d0511362aeb
        [2]: https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#ga4ff0f3318642c4f469d0e11f242f3b6c

        @arg image image to process
        """
        image = cv2.erode(image, None, iterations = 4)
        return cv2.dilate(image, None, iterations = 4)

    def contours(self, image: Image) -> np.intp:
        """
        Computes the contours / borders of the detected barcode and returns
        their edge points as np.intp

        @arg image image to process
        @returns contours
        """
        # returns 
        (cnts, _) = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = max(cnts, key=cv2.contourArea)

        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        return np.intp(box)

    def draw_box(self, image: Image, box: np.intp) -> Image:
        """
        Draws the contours onto the given image with the contours using the
        color set with self.border_color

        @arg image image to process
        @arg box contour points
        """
        return cv2.drawContours(image, [box], -1, self.border_color, 3)

    def process(self):
        """
        Apply barcode detection algorithm to the image located at self.image_path
        """
        og_img = cv2.imread(self.image_path)
        # original image
        # self.show_image(image)

        # image is already greyscale, no need to convert (or so i thought. I
        # got a cv2 error format mismatch, ig i have to greyscale this)
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

# used for checking if the script is included via an other script or it is ran
# directly
if __name__ == "__main__":
    flag = argparse.ArgumentParser()
    flag.add_argument("-p", "--path", help="Path to the input image", required=True)
    flag.add_argument("-c", "--color", help="Color of the border, fmt: r:g:b", required=False, default="0:255:0")
    args = flag.parse_args()
    bc = BarCodeDector(args.path, args.color)
    bc.process()
else:
    raise RuntimeError("Script intended to be ran, not included")
