# PyTorch-Onnx-Tensorrt
A set of tool which would make your life easier with Tensorrt and Onnxruntime for Yolov3.

## Requirements
1. Python 3
2. OpenCV
3. PyTorch
4. Onnx 1.4.1

## Downloading YoloV3 Configs and Weights
```
mkdir cfg
cd cfg 
wget https://raw.githubusercontent.com/pjreddie/darknet/f86901f6177dfc6116360a13cc06ab680e0c86b0/cfg/yolov3.cfg

mkdir weights
cd weights
wget https://pjreddie.com/media/files/yolov3.weights
```

## Running the detector Using Pytorch

```
python3 detect.py --cfg cfg/yolov3.cfg --weights weights/yolov3.weights 
```

## Generating the Onnx File

```
python3 create_onnx.py --reso 608
```