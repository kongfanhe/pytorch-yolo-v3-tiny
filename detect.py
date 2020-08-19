from models import YoloNet
from dataset import create_image, reset_random, colors
import cv2


def draw_bbox(img, x, y, w, h, color, t):
    x1, y1, x2, y2 = x - w/2, y - h/2, x + w/2, y + h/2
    res_x, res_y = img.shape[1], img.shape[0]
    x1, y1, x2, y2 = int(res_x * x1), int(res_y * y1), int(res_x * x2), int(res_y * y2)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, t)
    return img


def main():
    net: YoloNet = YoloNet(len(colors))
    net.load_from_file("saved.weights")
    reset_random()
    for n in range(30):
        img, _, _, _, _, _ = create_image(416)
        xs, ys, ws, hs, cs = net.predict(img, 0.1, 0.1)
        cv2.imshow("img raw", img)
        for x, y, w, h, c in zip(xs, ys, ws, hs, cs):
            co = colors[c]
            img = draw_bbox(img, x, y, w, h, [co[0]+50, co[1]+50, co[2]+50], 2)
        cv2.imshow("img", img)
        cv2.waitKey()
    print("done detect")


if __name__ == "__main__":
    main()
