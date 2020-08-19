
from dataset import generate_batch, reset_random, colors
from models import YoloNet as Net
import torch
import time
import numpy as np


def train_a_batch(net, optimizer, images, targets):
    net.train()
    net.zero_grad()
    features = net(images)
    loss = net.loss_fn(features, targets)
    loss.backward()
    optimizer.step()
    return loss.item()


def performance(net, x, y_true):
    net.eval()
    with torch.no_grad():
        y_pred = net(x)
        loss = net.loss_fn(y_pred, y_true).item()
    return loss


def train(net: Net, img_dim, epochs, bs, its, lr):
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    images_te, targets_te = generate_batch(bs, img_dim, net.strides, net.anchors)
    images_te = net.convert_images(images_te)
    targets_te = net.convert_targets(targets_te)
    loss_te_min =  performance(net, images_te, targets_te)
    for ep in range(epochs):
        reset_random()
        loss_tr = 0
        for it in range(its):
            images_tr, targets_tr = generate_batch(bs, img_dim, net.strides, net.anchors)
            images_tr = net.convert_images(images_tr)
            targets_tr = net.convert_targets(targets_tr)
            loss_tr = train_a_batch(net, optimizer, images_tr, targets_tr)
            print(ep, it, loss_tr)
        loss_te = performance(net, images_te, targets_te)
        time_str = time.strftime('%H:%M:%S')
        line_str = time_str + "," + str(loss_tr) + "," + str(loss_te)
        open("log.txt", "a").write(line_str + "\n")
        if loss_te < loss_te_min:
            net.save_to_file("saved.weights")
            loss_te_min = loss_te


def main():
    epochs, bs, its, gpu = 2, 32, 500, 0
    net: Net = Net(len(colors), gpu)
    img_dim = 416
    train(net, img_dim, epochs, bs, its, 1e-3)
    train(net, img_dim, epochs, bs, its, 1e-5)


if __name__ == "__main__":
    main()
