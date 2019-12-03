import os
import sys
import time

import tensorrt as trt
from PIL import Image
import numpy as np
import torch
import utils.calibrator as calibrator
import argparse


TRT_LOGGER = trt.Logger()

def build_engine_onnx(model_file , flag_int8 = 0):
	with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
		builder.max_workspace_size = 1<<30  #common.GiB(1)
		# Load the Onnx model and parse it in order to populate the TensorRT network.
		with open(model_file, 'rb') as model:
			parser.parse(model.read())

		if flag_int8:
			builder.int8_mode = True
			builder.int8_calibrator = calibrator.Yolov3EntropyCalibrator(data_dir="JpegImgs", cache_file='INT8CacheFile')
		return builder.build_cuda_engine(network)

def save_engine(engine, engine_dest_path):
    print('Engine:', engine)
    buf = engine.serialize()
    with open(engine_dest_path, 'wb') as f:
        f.write(buf)

def arg_parse(): 
	parser = argparse.ArgumentParser(description='YOLO v3 Video Detection Module')

	parser.add_argument("--onnx_file", dest = 'onnx_file', help = "Onnx File Name", type = str)  
	parser.add_argument("--trt_file", dest = 'trt_file', help = "Tensorrt Output File Name", default = "yolov3.trt", type = str)
	return parser.parse_args()

def main():
	args = arg_parse()

	onnx_file = args.onnx_file
	trt_file = args.trt_file
	engine = build_engine_onnx(onnx_file)
	save_engine(engine, trt_file)

if __name__ == '__main__':
	main()

