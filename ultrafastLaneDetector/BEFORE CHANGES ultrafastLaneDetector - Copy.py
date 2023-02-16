import cv2
import torch
import scipy.special
import numpy as np
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from enum import Enum
from scipy.spatial.distance import cdist

from ultrafastLaneDetector.model import parsingNet

lane_colors = [(0,0,255),(0,255,0),(255,0,0),(0,255,255)]

# tusimple_row_anchor = [ 64,  68,  72,  76,  80,  84,  88,  92,  96, 100, 104, 108, 112,
# 			116, 120, 124, 128, 132, 136, 140, 144, 148, 152, 156, 160, 164,
# 			168, 172, 176, 180, 184, 188, 192, 196, 200, 204, 208, 212, 216,
# 			220, 224, 228, 232, 236, 240, 244, 248, 252, 256, 260, 264, 268,
# 			272, 276, 280, 284]

# tusimple_row_anchor = list(range(116, 284, 3))
tusimple_row_anchor = list(range(64, 232, 3))
# tusimple_row_anchor = list(range(64, 176, 2))

print(len(tusimple_row_anchor))
print(tusimple_row_anchor)

culane_row_anchor = [121, 131, 141, 150, 160, 170, 180, 189, 199, 209, 219, 228, 238, 248, 258, 267, 277, 287]

center_line = [(640, 0), (640, 720)]
horizontal_line1 = [(0, 64), (1280, 64)]
horizontal_line2 = [(0, 284), (1280, 284)]

def get_x_y_values(point_list):
	x = [point[0] for point in point_list]
	y = [point[1] for point in point_list]
	return x, y

def draw_line_from_poly(draw_x, polynomial, image, color):
	draw_y = np.polyval(polynomial, draw_x)
	points_to_draw = (np.asarray([draw_x, draw_y]).T).astype(np.int32)
	cv2.polylines(image, [points_to_draw], False, color, 5)  # args: image, points, closed, color

def ignore_y_higher_than(x, y, y_cutoff):
	ret_x = []
	ret_y = []
	for x_, y_ in zip(x, y):
		if(y_ < y_cutoff):
			ret_x.append(x_)
			ret_y.append(y_)
	return ret_x, ret_y

def ignore_y_lower_than(x, y, y_cutoff):
	ret_x = []
	ret_y = []
	for x_, y_ in zip(x, y):
		if(y_ > y_cutoff):
			ret_x.append(x_)
			ret_y.append(y_)
	return ret_x, ret_y


def calculate_polynomial_moving_average(polynomial_array, moving_average):
	#step one: trim array to only required size only	
	if(len(polynomial_array) < moving_average):
		return [0,0,0,0,0,0,0]
	polynomial_array = polynomial_array[-moving_average:]

	#calculate moving average
	a = [point[0] for point in polynomial_array]
	b = [point[1] for point in polynomial_array]
	c = [point[2] for point in polynomial_array]
	d = [point[3] for point in polynomial_array]

	return [np.mean(a), np.mean(b), np.mean(c), np.mean(d)]


class ModelType(Enum):
	TUSIMPLE = 0
	CULANE = 1

global right_lane_polynomial_arr, left_lane_polynomial_arr
right_lane_polynomial_arr = []
left_lane_polynomial_arr = []

class ModelConfig():

	def __init__(self, model_type):

		if model_type == ModelType.TUSIMPLE:
			self.init_tusimple_config()
		else:
			self.init_culane_config()

	def init_tusimple_config(self):
		self.img_w = 1280
		self.img_h = 720
		self.row_anchor = tusimple_row_anchor
		self.griding_num = 100
		self.cls_num_per_lane = 56

	def init_culane_config(self):
		self.img_w = 1640
		self.img_h = 590
		self.row_anchor = culane_row_anchor
		self.griding_num = 200
		self.cls_num_per_lane = 18

class UltrafastLaneDetector():

	def __init__(self, model_path, model_type=ModelType.TUSIMPLE, use_gpu=False):

		self.use_gpu = use_gpu

		# Load model configuration based on the model type
		self.cfg = ModelConfig(model_type)

		# Initialize model
		self.model = self.initialize_model(model_path, self.cfg, use_gpu)

		# Initialize image transformation
		self.img_transform = self.initialize_image_transform()

	@staticmethod
	def initialize_model(model_path, cfg, use_gpu):

		# Load the model architecture
		net = parsingNet(pretrained = False, backbone='18', cls_dim = (cfg.griding_num+1,cfg.cls_num_per_lane,4),
						use_aux=False) # we dont need auxiliary segmentation in testing


		# Load the weights from the downloaded model
		if use_gpu:
			net = net.cuda()
			state_dict = torch.load(model_path, map_location='cuda')['model'] # CUDA
		else:
			state_dict = torch.load(model_path, map_location='cpu')['model'] # CPU

		compatible_state_dict = {}
		for k, v in state_dict.items():
			if 'module.' in k:
				compatible_state_dict[k[7:]] = v
			else:
				compatible_state_dict[k] = v

		# Load the weights into the model
		net.load_state_dict(compatible_state_dict, strict=False)
		net.eval()

		return net

	@staticmethod
	def initialize_image_transform():
		# Create transfom operation to resize and normalize the input images
		img_transforms = transforms.Compose([
			transforms.Resize((288, 800)),
			transforms.ToTensor(),
			transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
		])

		return img_transforms

	def detect_lanes(self, image, draw_points=True):

		input_tensor = self.prepare_input(image)

		# Perform inference on the image
		output = self.inference(input_tensor)

		# Process output data
		self.lanes_points, self.lanes_detected = self.process_output(output, self.cfg)

		# Draw depth image
		visualization_img = self.draw_lanes(image, self.lanes_points, self.lanes_detected, self.cfg, draw_points)
        
        #here you can return second argument (is lane crossed - than is left / right lane crossed)
        # You need to understand what stays behind self.lanes_points and self.lanes_detected
        # Return boolean which will trigger some action
        # But first try to understand it
		return visualization_img

	def prepare_input(self, img):
		# Transform the image for inference
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		img_pil = Image.fromarray(img)
		input_img = self.img_transform(img_pil)
		input_tensor = input_img[None, ...]

		if self.use_gpu:
			input_tensor = input_tensor.cuda()

		return input_tensor

	def inference(self, input_tensor):
		with torch.no_grad():
			output = self.model(input_tensor)

		return output

	@staticmethod
	def process_output(output, cfg):		
		# Parse the output of the model
		processed_output = output[0].data.cpu().numpy()
		processed_output = processed_output[:, ::-1, :]
		prob = scipy.special.softmax(processed_output[:-1, :, :], axis=0)
		idx = np.arange(cfg.griding_num) + 1
		idx = idx.reshape(-1, 1, 1)
		loc = np.sum(prob * idx, axis=0)
		processed_output = np.argmax(processed_output, axis=0)
		loc[processed_output == cfg.griding_num] = 0
		processed_output = loc


		col_sample = np.linspace(0, 800 - 1, cfg.griding_num)
		col_sample_w = (col_sample[1] - col_sample[0])

		lanes_points = []
		lanes_detected = []

		max_lanes = processed_output.shape[1]
		for lane_num in range(max_lanes):
			lane_points = []
			# Check if there are any points detected in the lane
			if np.sum(processed_output[:, lane_num] != 0) > 2:

				lanes_detected.append(True)

				# Process each of the points for each lane
				for point_num in range(processed_output.shape[0]):
					if processed_output[point_num, lane_num] > 0:
						lane_point = [
							int(processed_output[point_num, lane_num] * col_sample_w * cfg.img_w / 800) - 1, 
							int(cfg.img_h * (cfg.row_anchor[cfg.cls_num_per_lane-1-point_num]/235)) - 1 ]
						lane_points.append(lane_point)
			else:
				lanes_detected.append(False)

			lanes_points.append(lane_points)
		return np.array(lanes_points, dtype=object), np.array(lanes_detected, dtype=object)


	@staticmethod
	def draw_lanes(input_img, lanes_points, lanes_detected, cfg, draw_points=True):
		# Write the detected line points in the image
		visualization_img = cv2.resize(input_img, (cfg.img_w, cfg.img_h), interpolation = cv2.INTER_AREA)

		# # Draw a mask for the current lane
		# if(lanes_detected[1] and lanes_detected[2]):
		# 	lane_segment_img = visualization_img.copy()
		# 	cv2.fillPoly(lane_segment_img, pts = [np.vstack((lanes_points[1],np.flipud(lanes_points[2])))], color =(255,191,0))
		# 	visualization_img = cv2.addWeighted(visualization_img, 0.7, lane_segment_img, 0.3, 0)
		
		 
		if(lanes_detected[1]):
			x,y = get_x_y_values(lanes_points[1])
			#TODO(adam-kuzanski): here you can cut y axis if needed

			left_lane_polynomial = np.polyfit(x, y, 3)
			left_lane_polynomial_arr.append(left_lane_polynomial)

			final_poly = calculate_polynomial_moving_average(left_lane_polynomial_arr, 60)
			# print(final_poly)
			# print(type(final_poly))
			draw_line_from_poly(x, final_poly, visualization_img, (80, 255, 20))

			# global left_lane_polynomial 
			# left_lane_polynomial = np.polyfit(x, y, 2)
			# draw_line_from_poly(x, left_lane_polynomial, visualization_img, (100, 55, 50))

		if(lanes_detected[2]):
			x,y = get_x_y_values(lanes_points[2])
			
			x_new,y_new = ignore_y_higher_than(x, y, 600)

			#the lower value - the higher line will reach
			x_new, y_new = ignore_y_lower_than(x_new, y_new, 320)

			right_lane_polynomial = np.polyfit(x_new, y_new, 3)
			right_lane_polynomial_arr.append(right_lane_polynomial)

			final_poly = calculate_polynomial_moving_average(right_lane_polynomial_arr, 60)
			# print(final_poly)
			# print(type(final_poly))
			draw_line_from_poly(x_new, final_poly, visualization_img, (255, 55, 20))
				
			


		# if(draw_points):
		# 	for lane_num,lane_points in enumerate(lanes_points):
		# 		for lane_point in lane_points:
		# 			#print("Lane Number:{} \t:Lane Point{}".format(lane_num, lane_point))
		# 			cv2.circle(visualization_img, (lane_point[0],lane_point[1]), 3, lane_colors[lane_num], -1)
			
		# cv2.line(visualization_img, center_line[0], center_line[1], (0,255,0), 3) 
		# cv2.line(visualization_img, horizontal_line1[0], horizontal_line1[1], (20,255,0), 3) 
		# cv2.line(visualization_img, horizontal_line2[0], horizontal_line2[1], (100,255,100), 3) 
		

		return visualization_img


