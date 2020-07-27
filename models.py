import torch.nn as nn
import torch
import numpy as np

cross_entropy = nn.CrossEntropyLoss(reduction="sum")


def convolution(in_filters, out_filters, size, stride, pad, bn, ac):
    module = nn.Sequential()
    padding = (size - 1) // 2 if pad else 0
    c = nn.Conv2d(in_filters, out_filters, size, stride, padding, bias=not bn)
    module.add_module("cv", c)
    if bn:
        module.add_module("bn", nn.BatchNorm2d(out_filters))
    if ac:
        module.add_module("ac", nn.LeakyReLU(0.1, inplace=True))
    return module


def yolo_head(feature, anchor, device):
    # feature: [n_batch, n_anchor * (5 + n_class), n_grid, n_grid]
    # pred: [n_batch, n_grid, n_grid, n_anchor, (5 + n_class)]

    na = len(anchor)
    nb = feature.size(0)
    ng = feature.size(2)
    grids = np.linspace(0, (ng - 1) / ng, ng)
    gy, gx = np.meshgrid(grids, grids)
    cx = torch.tensor(gx).view(1, ng, ng, 1).repeat(nb, 1, 1, na).float().to(device)
    cy = torch.tensor(gy).view(1, ng, ng, 1).repeat(nb, 1, 1, na).float().to(device)
    scale = torch.tensor(anchor).repeat(1, ng, ng, 1).view(1, ng, ng, na, 2).float().to(device)
    pw = scale[:, :, :, :, 0]
    ph = scale[:, :, :, :, 1]

    pred = feature.permute(0, 2, 3, 1).contiguous().view(nb, ng, ng, na, -1)
    pred[:, :, :, :, 0] = torch.sigmoid(pred[:, :, :, :, 0])
    pred[:, :, :, :, 1] = torch.sigmoid(pred[:, :, :, :, 1]) / ng + cx
    pred[:, :, :, :, 2] = torch.sigmoid(pred[:, :, :, :, 2]) / ng + cy
    pred[:, :, :, :, 3] = torch.exp(pred[:, :, :, :, 3]) * pw
    pred[:, :, :, :, 4] = torch.exp(pred[:, :, :, :, 4]) * ph
    pred[:, :, :, :, 5:] = torch.sigmoid(pred[:, :, :, :, 5:])

    return pred


def yolo_loss(feature, target):

    # feature: [n_batch, n_anchor * (5 + n_class), n_grid, n_grid]
    # target: [n_batch, n_grid, n_grid, n_anchor, (5 + n_class)]

    nb, ng, na = target.size(0), target.size(1), target.size(3)
    p = feature.permute(0, 2, 3, 1).contiguous().view(nb, ng, ng, na, -1)
    p[:, :, :, :, 0:3] = torch.sigmoid(p[:, :, :, :, 0:3])
    p[:, :, :, :, 5:] = torch.sigmoid(p[:, :, :, :, 5:])
    obj_mask = target[:, :, :, :, 0]
    loss_obj = torch.sum((p - target)[:, :, :, :, 0] ** 2 * obj_mask)
    loss_no_obj = torch.sum((p - target)[:, :, :, :, 0] ** 2 * (1 - obj_mask))
    loss_box = torch.sum((p - target)[:, :, :, :, 1:5] ** 2 * obj_mask[:, :, :, :, None])
    loss_cls = torch.sum((p - target)[:, :, :, :, 5:] ** 2 * obj_mask[:, :, :, :, None])
    loss = loss_obj + 0.1 * loss_no_obj + loss_box + loss_cls
    return loss


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


def non_max_supress(x, y, w, h, th_iou):
    n = 0
    while len(x) > n:
        m = bbox_iou(x[n], y[n], w[n], h[n], x[:], y[:], w[:], h[:]) < th_iou
        m[:n + 1] = True
        x, y, w, h = x[m], y[m], w[m], h[m]
        n = n + 1
    return x, y, w, h


def get_device(gpu):
    if gpu >= 0 and torch.cuda.is_available():
        return torch.device("cuda:" + str(gpu))
    return torch.device("cpu")


class YoloNet(nn.Module):

    anchors = [[[0.02, 0.03], [0.06, 0.06], [0.09, 0.14]], [[0.19, 0.2], [0.32, 0.4], [0.83, 0.77]]]
    strides = [32, 16]

    def __init__(self, n_class, gpu=-1):
        super().__init__()
        self.cv_1 = convolution(3, 16, 3, 1, True, True, True)
        self.mp_1 = nn.MaxPool2d(2, 2)
        self.cv_2 = convolution(16, 32, 3, 1, True, True, True)
        self.mp_2 = nn.MaxPool2d(2, 2)
        self.cv_3 = convolution(32, 64, 3, 1, True, True, True)
        self.mp_3 = nn.MaxPool2d(2, 2)
        self.cv_4 = convolution(64, 128, 3, 1, True, True, True)
        self.mp_4 = nn.MaxPool2d(2, 2)
        self.cv_5 = convolution(128, 256, 3, 1, True, True, True)
        self.cv_5a = convolution(128, 256, 3, 1, True, True, True)
        self.mp_5 = nn.MaxPool2d(2, 2)
        self.cv_6 = convolution(256, 512, 3, 1, True, True, True)
        self.zp_6 = nn.ZeroPad2d((0, 1, 0, 1))
        self.mp_6 = nn.MaxPool2d(2, 1)
        self.cv_7 = convolution(512, 1024, 3, 1, True, True, True)
        self.cv_8 = convolution(1024, 256, 1, 1, True, True, True)
        self.cv_8a = convolution(1024, 256, 1, 1, True, True, True)
        self.cv_9 = convolution(256, 512, 3, 1, True, True, True)
        self.cv_10 = convolution(512, len(self.anchors[0]) * (5 + n_class), 1, 1, True, False, False)
        self.cv_11 = convolution(256, 128, 1, 1, True, True, True)
        self.up_11 = nn.Upsample(scale_factor=2)
        self.cv_13 = convolution(384, 256, 3, 1, True, True, True)
        self.cv_14 = convolution(256, len(self.anchors[1]) * (5 + n_class), 1, 1, True, False, False)
        self.device = get_device(gpu)
        self.to(self.device)
        self.n_class = n_class

    def forward(self, x):
        x = self.mp_1(self.cv_1(x))
        x = self.mp_2(self.cv_2(x))
        x = self.mp_3(self.cv_3(x))
        x_4 = self.mp_4(self.cv_4(x))
        x_5 = self.mp_5(self.cv_5(x_4))
        x_7 = self.cv_7(self.mp_6(self.zp_6(self.cv_6(x_5))))
        x_10 = self.cv_10(self.cv_9(self.cv_8(x_7)))
        x_5a = self.cv_5a(x_4)
        x_8a = self.cv_8a(x_7)
        x_11 = self.up_11(self.cv_11(x_8a))
        x_12 = torch.cat((x_11, x_5a), dim=1)
        x_14 = self.cv_14(self.cv_13(x_12))
        features = [x_10, x_14]
        return features

    def save_to_file(self, file):
        state = self.state_dict()
        torch.save(state, file)

    def load_from_file(self, file):
        state = torch.load(file, map_location=self.device)
        self.load_state_dict(state)

    @staticmethod
    def loss_fn(features, targets):
        # one feature: [n_batch, n_anchor * (5 + n_class), n_grid, n_grid]
        # one target: [n_batch, n_grid, n_grid, n_anchor, (5 + n_class)]
        loss = 0
        for (f, t) in zip(features, targets):
            loss += yolo_loss(f, t)
        return loss

    def convert_targets(self, targets):
        ts = []
        for t in targets:
            ts.append(torch.tensor(t).float().to(self.device))
        return ts

    def convert_images(self, images):
        # images: [n_batch, channel, width, height]
        return torch.tensor(images).float().to(self.device)

    def predict(self, img, th_pc, th_iou):
        self.eval()
        img = torch.tensor(img.transpose((2, 1, 0)).astype(float) / 255)
        features = self(img[None, :, :, :].float())
        p = np.empty((0, 5 + self.n_class))
        for (feature, anchor) in zip(features, self.anchors):
            pred = yolo_head(feature, anchor, self.device)
            _p = np.reshape(pred.detach().numpy(), (-1, pred.size(-1)))
            p = np.concatenate((p, _p), axis=0)
        p = p[p[:, 0] > th_pc, :]
        p = p[np.argsort(-p[:, 0]), :]
        cls = np.argmax(p[:, 5:], axis=1)
        xs, ys, ws, hs, cs = [], [], [], [], []
        for c in np.unique(cls):
            m = cls == c
            x, y, w, h = non_max_supress(p[m, 1], p[m, 2], p[m, 3], p[m, 4], th_iou)
            xs.extend(x)
            ys.extend(y)
            ws.extend(w)
            hs.extend(h)
            cs.extend([c] * len(x))
        return xs, ys, ws, hs, cs


def main():
    nb, nc, img_dim = 30, 3, 416
    net: YoloNet = YoloNet(nc)
    img = torch.tensor(np.zeros((nb, 3, img_dim, img_dim))).float()
    net.train()
    net.zero_grad()
    features = net(img)
    targets = []
    for (anchor, s) in zip(net.anchors, net.strides):
        g = img_dim // s
        targets.append(np.zeros((nb, g, g, len(anchor), 5 + nc)))
    targets = net.convert_targets(targets)
    loss = net.loss_fn(features, targets)
    print("loss ready ...")
    loss.backward()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    optimizer.step()
    print("done")


if __name__ == "__main__":
    main()
