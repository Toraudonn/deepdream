#!/usr/bin/env python3

import torch
import torch.nn as nn
from torchvision import models, transforms

import numpy as np
import scipy.ndimage as nd
import tqdm

from utils import deprocess, preprocess, clip


def get_base_model(at_layer):
    network = models.vgg19(pretrained=True)
    layers = list(network.features.children())
    model = nn.Sequential(*layers[: (at_layer + 1)])
    
    if torch.cuda.is_available():
        model = model.cuda()

    return model


def dream(img, model, iterations, lr):
    """ Updates the image to maximize outputs for n iterations """
    img = torch.tensor(img, requires_grad=True)
    
    for i in range(iterations):

        model.zero_grad()
        out = model(img)
        
        loss = out.norm()
        loss.backward()
        
        avg_grad = np.abs(img.grad.data.cpu().numpy()).mean()
        norm_lr = lr / avg_grad

        img.data += norm_lr * img.grad.data
        img.data = clip(img.data)
        img.grad.zero_()

    return img.detach().cpu().numpy()


def deep_dream(img, at_layer, iterations, lr, octave_scale, num_octaves):
    """ Main deep dream method """

    model = get_base_model(at_layer) 
    img = preprocess(img).unsqueeze(0).cpu().numpy()

    # Extract image representations for each octave
    octaves = [img]
    for _ in range(num_octaves - 1):
        octaves.append(
            nd.zoom(
                octaves[-1],
                (1, 1, 1 / octave_scale, 1 / octave_scale),
                order=1
            )
        )

    dreamed_img = None
    detail = np.zeros_like(octaves[-1])
    for octave, octave_base in enumerate(
        tqdm.tqdm(octaves[::-1], desc="Dreaming")
    ):
        if octave > 0:
            # Upsample detail to new octave dimension
            detail = nd.zoom(detail, np.array(octave_base.shape) / np.array(detail.shape), order=1)
        # Add deep dream detail from previous octave to new base
        _img = octave_base + detail
        # Get new deep dream image
        dreamed_img = dream(_img, model, iterations, lr)
        # Extract deep dream details
        detail = dreamed_img - octave_base
    if dreamed_img is not None:
        return deprocess(dreamed_img)
    else:
        return None