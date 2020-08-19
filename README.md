# Pytorch implementation of YOLO-v3-tiny

This repository provides an easy pytorch implementaion of YOLO-v3-tiny.
The dataset is images with color rectangles on black background. 
The images are generated with OPENCV on the fly. 


**No extra data download, plug and play.**


## How to use

1. Install Python >=3.6

2. Install necessary packages
```bash
    pip install -r requirements
```

3. Train a YOLO-v3-tiny model, and save weights to "saved.weights" file.
```bash
    python train.py
```

4. Test on dataset, with "saved.weights" file.
```bash
    python detect.py
```

## Test Results

The training process takes 20 minutes to converge. 
Our PC configuration is:

|  |  |
| --- | --- |
| CPU | I5 9400F |
| GPU | GTX-1050Ti |
| RAM | 8GB |


The loss function on test dataset drops from ~2000 to ~10, and we stopped there.
The learning curve (only test loss) is plotted below:

![](https://wx2.sinaimg.cn/mw690/008b8Ivhly1ghw3g4gxoej30hs0dc74l.jpg)

With "saved.weights" file, we can predict on new data:

![](https://wx2.sinaimg.cn/small/008b8Ivhgy1ghvjhntdvvj30eg0ega9x.jpg)
![](https://wx3.sinaimg.cn/small/008b8Ivhgy1ghvjhlf3c8j30eg0egdfo.jpg)


