# barcode detection

> detect barcodes in images and draw a border with a custom color around them

```shell
$ python3 barcode-detector.py --path example.tif --color "255:0:255"
```
![Figure_1](https://github.com/xNaCly/barcode-detection/assets/47723417/3e8359f6-46b8-4c25-a7af-692f56fee2c4)




## Usage

```
usage: barcode-detector.py [-h] -p PATH [-c COLOR]

options:
  -h, --help              show this help message and exit
  -p PATH, --path PATH    Path to the input image
  -c COLOR, --color COLOR Color of the border, fmt: r:g:b
```

## Inner workings

The barcode detector works by applying several filters to an image and
extracting the contours of the barcodes. If a barcode could not be detected
consider tweaking constants in the source file.

The algorithm is made up of the following filters and stages:

1. sharr gradient in x and y direction
2. subtract y from x gradient for finding barcode region
3. blur and threshold
4. Closing kernel
5. Series of dilations and erosions
6. Largest contour should be the barcode
