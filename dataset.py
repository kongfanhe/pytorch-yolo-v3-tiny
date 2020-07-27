import numpy as np
import cv2
from random import uniform
import random

colors = [[0, 0, 70], [60, 60, 0]]


def reset_random():
    random.seed(0)


def random_xywh():
    w = uniform(0.2, 0.7)
    h = uniform(0.2, 0.7)
    x = uniform(0.1 + w / 2, 0.9 - w / 2)
    y = uniform(0.1 + h / 2, 0.9 - h / 2)
    return x, y, w, h


def fill_rectangle(img, x, y, w, h, color):
    x1, y1, x2, y2 = x - w / 2, y - h / 2, x + w / 2, y + h / 2
    res_x, res_y = img.shape[1], img.shape[0]
    x1, y1 = int(res_x * x1), int(res_y * y1)
    x2, y2 = int(res_x * x2), int(res_y * y2)
    img[y1:y2, x1:x2] = img[y1:y2, x1:x2] + color


def create_image(image_dim):
    img = np.zeros((image_dim, image_dim, 3), np.uint8)
    xs, ys, ws, hs, cs = [], [], [], [], []
    n_shapes = random.choice([1, 2])
    classes = random.sample(range(len(colors)), n_shapes)
    for c in classes:
        x, y, w, h = random_xywh()
        fill_rectangle(img, x, y, w, h, colors[c])
        xs.append(x)
        ys.append(y)
        ws.append(w)
        hs.append(h)
        cs.append(c)
    return img, xs, ys, ws, hs, cs


def bbox_iou(xa, ya, wa, ha, xb, yb, wb, hb):
    x1, y1, x2, y2 = xa - wa / 2, ya - ha / 2, xa + wa / 2, ya + ha / 2
    x3, y3, x4, y4 = xb - wb / 2, yb - hb / 2, xb + wb / 2, yb + hb / 2
    x5 = np.maximum(x1, x3)
    y5 = np.maximum(y1, y3)
    x6 = np.minimum(x2, x4)
    y6 = np.minimum(y2, y4)
    w = np.maximum(x6 - x5, 0)
    h = np.maximum(y6 - y5, 0)
    inter_area = w * h
    b1_area = (x2 - x1) * (y2 - y1)
    b2_area = (x4 - x3) * (y4 - y3)
    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)
    return iou


def yolo_targets(xs, ys, ws, hs, cs, grids, anchors):
    targets = []
    for n, (g, anchor) in enumerate(zip(grids, anchors)):
        targets.append(np.zeros((g, g, len(anchor), 5 + len(colors))))
        anchors[n] = np.array(anchor)
    for x, y, w, h, c in zip(xs, ys, ws, hs, cs):
        _iou_best, i0, j0 = 0, 0, 0
        for i in range(len(anchors)):
            _iou = bbox_iou(0, 0, w, h, 0, 0, anchors[i][:, 0], anchors[i][:, 1])
            _j = np.argmax(_iou)
            if _iou[_j] > _iou_best:
                _iou_best = _iou[_j]
                i0 = i
                j0 = _j
        g, w0, h0 = grids[i0], anchors[i0][j0, 0], anchors[i0][j0, 1]
        nx = int(x // (1 / g))
        ny = int(y // (1 / g))
        cls = [1 if i == c else 0 for i in range(len(colors))]
        targets[i0][nx, ny, j0, :] = [1, x * g - nx, y * g - ny, np.log(w / w0), np.log(h / h0)] + cls
    return targets


def generate_batch(batch_size, image_dim, strides, anchors):
    images, _targets = [], [[] for _ in range(len(strides))]
    grids = [image_dim // s for s in strides]
    for _ in range(batch_size):
        img, xs, ys, ws, hs, cs = create_image(image_dim)
        ts = yolo_targets(xs, ys, ws, hs, cs, grids, anchors)
        img = img.transpose((2, 1, 0)).astype(float) / 255
        images.append(img)
        for n, t in enumerate(ts):
            _targets[n].append(t)
    targets = []
    for t in _targets:
        targets.append(np.array(t))
    images = np.array(images)
    return images, targets


def main():
    reset_random()
    anchors = [[[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]], [[0.4, 0.4], [0.5, 0.5], [0.6, 0.6]]]
    strides = [32, 16]
    image_dim = 416
    bs = 30
    for n in range(10):
        images, targets = generate_batch(bs, image_dim, strides, anchors)
        img = (images[0, :, :, :].transpose((2, 1, 0)) * 255).astype(np.uint8)
        print(n, images.shape, len(targets), targets[0].shape, targets[1].shape)
        cv2.imshow("img", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
