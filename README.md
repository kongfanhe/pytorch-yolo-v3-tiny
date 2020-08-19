# Pytorch implementation of YOLO-v3-tiny

This repository provides an easy pytorch implementaion of YOLO-v3-tiny.
The dataset is a set of images with color rectangles on black background. 
The images are generated with OPENCV. You do not need to download any extra dataset, 
just **plug and play**.


## How to play

1. Install Python >=3.6

2. Install necessary library
```bash
    pip install -r requirements
```

3. Train a YOLO-v3-tiny model, and obtain weights file "saved.weights"
```bash
    python train.py
```

4. Test on dataset, using "saved.weights"
```bash
    python detect.py
```

## Test Results

The training process takes around one hour on GTX-1050Ti. Below are some prediction on new data:

![](https://wx2.sinaimg.cn/small/008b8Ivhgy1ghvjhntdvvj30eg0ega9x.jpg)
![](https://wx3.sinaimg.cn/small/008b8Ivhgy1ghvjhlf3c8j30eg0egdfo.jpg)


