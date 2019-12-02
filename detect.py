import sys
import time
from PIL import Image, ImageDraw
from utils.util import *
from darknet import Darknet
import argparse

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

def detect_image(frame,model, inp_dim = 416, CUDA = 1):

	img, dim = prep_image_new(frame, inp_dim)
	# img = torch.from_numpy(frame.transpose(2,0,1)).float().div(255.0).unsqueeze(0)

	im_dim = torch.FloatTensor(dim).repeat(1,2)

	if CUDA:
		im_dim = im_dim.cuda()

	with torch.no_grad():   
		output = model(Variable(img), CUDA)

	# output = write_results_batch(output, 0.3, 80, nms = True, nms_conf = 0.25)
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

def detect(cfgfile, weightfile, imgfile):

	model = Darknet(cfgfile)
	model.load_weights(weightfile)
	model.net_info["height"] = 416
	model.cuda()
	model.eval()

	img = cv2.imread(imgfile)
	img_torch = torch.from_numpy(img).float()

	car_boxes = detect_image(img_torch,model)

	for i in car_boxes:
		cv2.rectangle(img,(i[0],i[1]),(i[2],i[3]),(255,255,255),2)
		
	cv2.imwrite("result.png", img)

def create_onnx(cfgfile, weightfile, reso = 416):
	model = Darknet(cfgfile)
	# model.load_weights(weightfile)
	model.load_state_dict(torch.load("yolo.pth"))
	model.cuda()
	model.eval()

	dummy_input = Variable(torch.randn(1, 3, reso, reso)).to(0)
	# dummy_input = Variable(torch.randn(1, 3, reso, reso))
	torch.onnx.export(model, dummy_input , "yolo.onnx")


def arg_parse(): 
	parser = argparse.ArgumentParser(description='YOLO v3 Video Detection Module')

	parser.add_argument("--annot", dest="annotation_path",help="no help needed", default ="output_CAM_PTZ.npy", type= str)
	parser.add_argument("--confidence", dest = "confidence", help = "Object Confidence to filter predictions", default = 0.20)
	parser.add_argument("--nms_thresh", dest = "nms_thresh", help = "NMS Threshhold", default = 0.25)
	parser.add_argument("--img", dest = "img", default = "test/1.png", help = "Image File")
	parser.add_argument("--cfg", dest = 'cfgfile', help = "Config file",
						default = "cfg/yolov3.cfg", type = str)  ## yolov3.cfg  yolov3.weights  ## yolov3_cls_1.cfg 
	parser.add_argument("--weights", dest = 'weightsfile', help = "weightsfile",
						default = "yolov3.weights", type = str)  ## yolov3_cls_1_200.weights   ## yolov3_cls_1_final.weights
	parser.add_argument("--reso", dest = 'reso', help = 
						"Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
						default = "416", type = int)
	return parser.parse_args()

if __name__ == '__main__':
	args = arg_parse()
	
	cfgfile = args.cfgfile
	weightfile = args.weightsfile
	imgfile = args.img
	
	detect(cfgfile, weightfile, imgfile)
