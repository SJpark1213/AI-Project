import sys
import cv2
import random as rand
import numpy as np
import GraphOperator as go
import os


def generate_image(ufset, width, height):
    random_color = lambda: (int(rand.random() * 255), int(rand.random() * 255), int(rand.random() * 255))
    color = [random_color() for i in range(width * height)]

    save_img = np.zeros((height, width, 3), np.uint8)

    for y in range(height):
        for x in range(width):
            color_idx = ufset.find(y * width + x)
            save_img[y, x] = color[color_idx]

    return save_img

sigma = 0.999
k = 500
min_size = 100

list_dir = os.listdir('./crack')

for name in list_dir:
    filename = os.path.join('./crack', name)

    img = cv2.imread(filename)
    float_img = np.asarray(img, dtype=float)

    gaussian_img = cv2.GaussianBlur(float_img, (5, 5), sigma)
    b, g, r = cv2.split(gaussian_img)
    smooth_img = (r, g, b)

    height, width, channel = img.shape
    graph = go.build_graph(smooth_img, width, height)

    weight = lambda edge: edge[2]
    sorted_graph = sorted(graph, key=weight)

    ufset = go.segment_graph(sorted_graph, width * height, k)
    ufset = go.remove_small_component(ufset, sorted_graph, min_size)

    save_img = generate_image(ufset, width, height)

    save_name = os.path.join('./crack_result1', name)
    cv2.imwrite(save_name, save_img)