import sys
import time
from PIL import Image, ImageDraw
from utils.util import *
from darknet import Darknet
import argparse
import onnxruntime


def resize2d(img, size):
	return (F.adaptive_avg_pool2d(Variable(img,volatile=True), size)).data

def prep_image_new(frame, inp_dim):
	orig_im = frame
	dim = orig_im.shape[1], orig_im.shape[0]

	max_dim = max(dim)
	w_pad = int(abs(dim[0] - max_dim)/2)
	h_pad = int(abs(dim[1] - max_dim)/2)

	img_torch = frame
	img_torch = img_torch.cuda()
	img_torch_new = img_torch.transpose(0,2)     ## Swap depth with 0 idx
	img_torch_new = img_torch_new.transpose(1,2) ## Swap other two dims

	img_pad = torch.nn.functional.pad(img_torch_new,(w_pad, w_pad, h_pad,h_pad),value=128)
	img_resize = resize2d(img_pad,inp_dim)
	img_resize = img_resize.div(255.0).unsqueeze(0)

	return img_resize , dim 

def bbox_filtering(output, inp_dim, im_dim, CUDA = True):

	output = write_results(output, 0.3, 80, nms = True, nms_conf = 0.25)
	im_dim = im_dim.repeat(output.size(0), 1)

	scaling_factor = torch.min(inp_dim/im_dim,1)[0].view(-1,1)
	
	output[:,[1,3]] -= (inp_dim - scaling_factor*im_dim[:,0].view(-1,1))/2
	output[:,[2,4]] -= (inp_dim - scaling_factor*im_dim[:,1].view(-1,1))/2
	
	output[:,1:5] /= scaling_factor

	output[:, [1,3]] = torch.clamp(output[:, [1,3]], 0.0, im_dim[0,0])
	output[:, [2,4]] = torch.clamp(output[:, [2,4]], 0.0, im_dim[0,1])

	np_output = (np.array(output.cpu()))

	np_output = np_output[:, 1:5]
	np_output = np_output.astype(int)
	np_output = np_output.tolist()

	return np_output

def detect_image_onnx(frame, session, CUDA = True):
	anchors_arr = [[(116, 90), (156, 198), (373, 326)], [(30, 61), (62, 45), (59, 119)], [(10, 13), (16, 30), (33, 23)]]
	dim = (frame.shape[1], frame.shape[0])

	inp_dim = session.get_inputs()[0].shape[2]
	first_input_name = session.get_inputs()[0].name

	dat = cv2.dnn.blobFromImage(frame, 1 / 255.0, (inp_dim, inp_dim),swapRB=True, crop=False)

	dict_input = {first_input_name : dat.astype(np.float32)}
	results = session.run([],dict_input)
	
	write = 0
	for i,x in enumerate(results):
		x = torch.from_numpy(x).to(0)
		x = predict_transform(x, inp_dim, anchors_arr[i], 80, CUDA = CUDA)
		if not write:
			output = x
			write = 1
		else:
			output = torch.cat((output, x), 1)

	im_dim = torch.FloatTensor(dim).repeat(1,2).to(0)	
	np_output = bbox_filtering(output, inp_dim, im_dim ,CUDA = CUDA)

	return np_output


def detect_image_pytorch(frame,model, CUDA = True):
	inp_dim = int(model.net_info["height"])
	frame = torch.from_numpy(frame).float()
	img, dim = prep_image_new(frame, inp_dim)
	im_dim = torch.FloatTensor(dim).repeat(1,2)

	if CUDA:
		im_dim = im_dim.cuda()

	with torch.no_grad():   
		output = model(Variable(img), CUDA)

	# output = write_results_batch(output, 0.3, 80, nms = True, nms_conf = 0.25)
	np_output = bbox_filtering(output, inp_dim, im_dim)

	return np_output

def detect(imgfile, model, onnx_flag = False):

	img = cv2.imread(imgfile)
	img_torch = torch.from_numpy(img).float()

	# for i in range(10):
	# 	t1 = time.time()

	if onnx_flag:
		car_boxes = detect_image_onnx(img, model)
	else:
		car_boxes = detect_image_pytorch(img,model)

	# t2 = time.time()
	# total_time = (t2 - t1)*1000
	# print("Total Time : ", total_time)

	for i in car_boxes:
		cv2.rectangle(img,(i[0],i[1]),(i[2],i[3]),(0,0,0),2)
		
	cv2.imwrite("result.png", img)

def arg_parse(): 
	parser = argparse.ArgumentParser(description='YOLO v3 Video Detection Module')

	parser.add_argument("--annot", dest="annotation_path",help="no help needed", default ="output_CAM_PTZ.npy", type= str)
	parser.add_argument("--confidence", dest = "confidence", help = "Object Confidence to filter predictions", default = 0.20)
	parser.add_argument("--nms_thresh", dest = "nms_thresh", help = "NMS Threshhold", default = 0.25)
	parser.add_argument("--img", dest = "img", default = "test/1.png", help = "Image File")
	parser.add_argument("--cfg", dest = 'cfgfile', help = "Config file",
						default = "cfg/yolov3.cfg", type = str)  
	parser.add_argument("--weights", dest = 'weightsfile', help = "weightsfile",
						default = "weights/yolov3.weights", type = str) 
	parser.add_argument("--use_onnx", dest = 'use_onnx', help = "Inference Using OnnxRuntime",
						default = False, type = bool) 
	parser.add_argument("--onnx_file", dest = 'onnx_file', help = "Onnx File Path",
						default = "yolov3.onnx", type = str)
	return parser.parse_args()

if __name__ == '__main__':
	args = arg_parse()
	
	cfgfile = args.cfgfile
	weightfile = args.weightsfile
	imgfile = args.img
	onnx_flag = args.use_onnx
	onnx_file = args.onnx_file

	if onnx_flag:
		session = onnxruntime.InferenceSession(onnx_file)
		session.get_modelmeta()
		detect(imgfile, session, onnx_flag)
	else:
		model = Darknet(cfgfile)
		model.load_weights(weightfile)
		model.cuda()
		model.eval()
		detect(imgfile, model)
