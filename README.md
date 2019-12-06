# PyTorch-Onnx-Tensorrt
A set of tool which would make your life easier with Tensorrt and Onnxruntime for Yolov3.

## Requirements
1. Python 3
2. OpenCV
3. PyTorch
4. Onnx 1.4.1
5. Onnxruntime
6. Tensorrt

I would Highly Recommend setting up a Nvidia Deepstream/Tensorrt Docker for these operations.

## Downloading YoloV3 Configs and Weights
```
mkdir cfg
cd cfg 
wget https://raw.githubusercontent.com/pjreddie/darknet/f86901f6177dfc6116360a13cc06ab680e0c86b0/cfg/yolov3.cfg

mkdir weights
cd weights
wget https://pjreddie.com/media/files/yolov3.weights
```

## Editing Config File
Inorder to Run the model in Pytorch or creating Onnx / Tensorrt File for different Input image Sizes ( 416, 608, 960 etc), you need to edit the Batch Size and Input image size in the config file - net info section.
```
batch=1
width=416
height=416
```

## Running the detector Using Pytorch

```
python3 detect.py --cfg cfg/yolov3.cfg --weights weights/yolov3.weights 
```

## Generating the Onnx File

```
python3 create_onnx.py --reso 416
```

## Running the detector Using ONNX
```
python3 detect.py --use_onnx True --onnx_file yolov3.onnx
```

## Generating the Tensorrt File

```
python3 create_trt_engine.py --onnx_file yolov3.onnx 
```
Creating the Tensorrt engine takes some time. So have some patience.
