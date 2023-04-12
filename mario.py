#%%
import torch
from torch import nn
from torchvision import transforms as T
from PIL import Image
import numpy as np
from pathlib import Path
from collections import deque
import random
import datetime
import os
import copy

import gym
from gym.spaces import Box
from gym.wrappers import FrameStack




# %%
import tensorflow as tf
from PIL import Image
img = Image.open("/Users/wonhyung64/Documents/증명서/증명사진.jpg")
img = tf.keras.utils.img_to_array(img)
small_img = tf.image.resize(img, (38, 30))
tf.keras.utils.array_to_img(small_img)
syns_img = tf.image.resize(small_img, (380, 300))
tf.keras.utils.array_to_img(syns_img)
model = tf.keras.applications.resnet.ResNet50(include_top=False)
org_feat = model(tf.expand_dims(tf.image.resize(img, (380, 300)), 0))
syns_feat = model(tf.expand_dims(syns_img, 0))

import matplotlib.pyplot as plt
plt.plot(org_feat[0])

square = 8
ix = 1
for _1 in range(square):
    for _2 in range(square):
        ax = plt.subplot(square, square, ix)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.imshow(syns_feat[0, :, :, ix-1], cmap="gray")
        ix += 1
plt.show()

