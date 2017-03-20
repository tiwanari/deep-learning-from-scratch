#!/usr/bin/env python3
import sys
sys.path.append('../..')
from dataset.mnist import load_mnist
import numpy as np
from PIL import Image


def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

(x_train, l_train), (x_test, l_text) \
    = load_mnist(flatten=True, normalize=False)

img = x_train[0]
label = l_train[0]
print(label)

print(img.shape)
img = img.reshape(28, 28)
print(img.shape)

img_show(img)
