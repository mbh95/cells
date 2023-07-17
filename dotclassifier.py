import csv
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.neighbors import BallTree
import argparse

"""
Usage:
python dotclassifier.py \
    --mask=_b3red_cp_masks.png \
    --outlines="b3red_cp_outlines updated.txt" \
    --dots="green_center_loc for b3.csv" \
    --composite="b3 Composite (RGB).tif" \
    --e=3
"""


def plot(outlines, on_dots, off_dots, bad_dots):
    img = np.asarray(Image.open('b3 Composite (RGB).tif'))
    plt.imshow(img)
    for i in range(len(outlines)):
        np_outline = np.array([*outlines[i]])
        plt.plot(np_outline[:, 0], np_outline[:, 1], color='r')
    for dot in on_dots:
        plt.scatter([dot[0]], [dot[1]], color='pink')
    for dot in off_dots:
        plt.scatter([dot[0]], [dot[1]], color='b')
    for dot in bad_dots:
        plt.scatter([dot[0]], [dot[1]], color='brown')
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        prog='DotClassifier',
        description='Classifies dots')
    parser.add_argument("-m", "--mask")
    parser.add_argument("-o", "--outlines")
    parser.add_argument("-d", "--dots")
    parser.add_argument("-c", "--composite")
    parser.add_argument("-e", "--epsilon", default=3)
    args = parser.parse_args()

    # Open mask file
    img = Image.open(args.mask)

    # Read in outlines
    outlinestxt = open(args.outlines)
    rawcoords = [[int(coord) for coord in line.split(",")]
                 for line in outlinestxt.readlines()]
    outlines = [[x for x in zip(rawoutline[::2], rawoutline[1::2])]
                for rawoutline in rawcoords]
    # Read in dots
    dotscsv = open(args.dots, newline="")
    dotsreader = csv.DictReader(dotscsv, skipinitialspace=True)
    dots = [(int(r["y"]), int(r["x"])) for r in dotsreader]

    border_pixels = set()

    for outline in outlines:
        for (x, y) in outline:
            border_pixels.add((x, y))
    tree = BallTree(np.array([*border_pixels]))

    epsilon = float(args.epsilon)
    on_dots = set()
    off_dots = set()
    bad_dots = set()

    dot_to_cell = {}
    cell_to_dot = {}

    for dot in dots:
        cell_num = img.getpixel(dot)
        if cell_num == 0:
            bad_dots.add(dot)
            continue
        dot_to_cell[dot] = cell_num
        if not cell_num in cell_to_dot:
            cell_to_dot[cell_num] = [dot]
        else:
            cell_to_dot[cell_num].append(dot)

        if len(cell_to_dot[dot_to_cell[dot]]) > 1:
            bad_dots.add(dot)
            continue
        nearest, indx = tree.query(np.array([[dot[0], dot[1]]]), k=1)
        if nearest[0] <= epsilon:
            on_dots.add(dot)
        else:
            off_dots.add(dot)
    print(f"on dots (pink): {len(on_dots)}")
    print(f"off dots (blue): {len(off_dots)}")
    print(f"total on+off dots: {len(on_dots)+len(off_dots)}")
    print(f"ignored dots (brown): {len(bad_dots)}")

    plot(outlines, on_dots, off_dots, bad_dots)


if __name__ == "__main__":
    main()
