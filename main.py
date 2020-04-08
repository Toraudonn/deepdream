#!/usr/bin/env python3

import os
import argparse

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from PIL import Image

from deep_dream import deep_dream


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--img_file", type=str, default="images/supermarket.jpg", help="path to input image")
    parser.add_argument("--iterations", default=20, help="number of gradient ascent steps per octave")
    parser.add_argument("--at_layer", default=27, type=int, help="layer at which we modify image to maximize outputs")
    parser.add_argument("--lr", default=0.01, help="learning rate")
    parser.add_argument("--octave_scale", default=1.4, help="image scale between octaves")
    parser.add_argument("--num_octaves", default=10, help="number of octaves")

    return parser.parse_args()


def main(
    img_file: str,
    iterations: int,
    at_layer: int,
    lr: float,
    octave_scale: float,
    num_octaves: int,
):
    # Load image
    img = np.asarray(Image.open(img_file))

    # Extract deep dream image
    dreamed_image = deep_dream(
        img,
        at_layer=at_layer,
        iterations=iterations,
        lr=lr,
        octave_scale=octave_scale,
        num_octaves=num_octaves,
    )

    # Save and plot image
    os.makedirs("outputs", exist_ok=True)
    filename = args.img_file.split("/")[-1]
    plt.figure(figsize=(20, 20))
    plt.imshow(dreamed_image)
    plt.imsave(f"outputs/dreamed_{filename}", dreamed_image)
    plt.show()


if __name__ == "__main__":
    args = get_args()
    main(**vars(args))

