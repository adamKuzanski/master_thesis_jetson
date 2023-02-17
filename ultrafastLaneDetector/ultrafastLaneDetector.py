import cv2
import torch
import math
import scipy.special
import numpy as np
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from enum import Enum
from scipy.spatial.distance import cdist

from shapely.geometry import LineString
import csv



from ultrafastLaneDetector.model import parsingNet

lane_colors = [(0,0,255),(0,255,0),(255,0,0),(0,255,255)]

tusimple_row_anchor = [ 64,  68,  72,  76,  80,  84,  88,  92,  96, 100, 104, 108, 112,
			116, 120, 124, 128, 132, 136, 140, 144, 148, 152, 156, 160, 164,
			168, 172, 176, 180, 184, 188, 192, 196, 200, 204, 208, 212, 216,
			220, 224, 228, 232, 236, 240, 244, 248, 252, 256, 260, 264, 268,
			272, 276, 280, 284]

# tusimple_row_anchor = list(range(116, 284, 3))
# tusimple_row_anchor = list(range(64, 232, 3))	
# tusimple_row_anchor = list(range(64, 176, 2))

# print(len(tusimple_row_anchor))
# print(tusimple_row_anchor)

culane_row_anchor = [121, 131, 141, 150, 160, 170, 180, 189, 199, 209, 219, 228, 238, 248, 258, 267, 277, 287]

LANE_CROSSING_FREEZER = 30
right_lane_freeze = 0
left_lane_freeze = 0

ROI_VERTICIES = np.array([[(200, 600),         		# bottom-left corner
							(500, 400),                	# top-left corner
							(700, 400),                	# top-right corner
							(1200, 600)]], dtype=np.int32) # bottom-right corner

def get_x_y_values(point_list):
	x = [point[0] for point in point_list]
	y = [point[1] for point in point_list]
	return x, y

def draw_line_from_poly(lane_points, polynomial, image, color):
	draw_x, y = get_x_y_values(lane_points)
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


def calculate_polynomial_moving_average(polynomial_array, moving_average = 5):
	#step one: trim array to only required size only	
	if(len(polynomial_array) < moving_average):
		return [0,0,0,0]
	polynomial_array = polynomial_array[-moving_average:]

	#calculate moving average
	a = [point[0] for point in polynomial_array]
	b = [point[1] for point in polynomial_array]
	c = [point[2] for point in polynomial_array]
	# d = [point[3] for point in polynomial_array]

	# return [np.mean(a), np.mean(b), np.mean(c), np.mean(d)]
	return [np.mean(a), np.mean(b), np.mean(c)]


def filter_roi_points(points, vertices):
	"""
	Filters a list of (x, y) points to only include the points within a specified ROI.

	Args:
		points: A list of (x, y) points.
		vertices: An array of vertices that define the ROI.

	Returns:
		A new list containing only the points within the ROI.
	"""
	# Create a mask with the same shape as the image
	mask = np.zeros((720, 1280), dtype=np.uint8)

	# Fill the polygon defined by the vertices with white color (255)
	cv2.fillPoly(mask, vertices, 255)	

	# Convert the list of points to a numpy array
	points_arr = np.array(points)

	# Convert the data type of the points array to floating-point
	points_arr = points_arr.astype(np.float32)

	# Convert the points to a list of tuples with floating-point values
	points_list = [tuple(point) for point in points_arr]

	# Use the cv2.pointPolygonTest() function to check which points are inside the ROI
	inside_mask = np.array([cv2.pointPolygonTest(vertices, point, False) for point in points_list]) >= 0

	# Return only the points that are inside the ROI
	return points_arr[inside_mask].tolist()



isRightLaneCrossed = False

class ModelType(Enum):
	TUSIMPLE = 0
	CULANE = 1

right_lane_polynomial_arr = []
left_lane_polynomial_arr = []
center_lane_polynomial_arr = []
distance_to_right_arr = []
distance_to_left_arr = []
delta_left_lane_arr = []
delta_right_lane_arr = []

def getAverageLanePolynomial(lane_points, lane_type, moving_average = 30):
	"""
	lane_points - points correspondig to a particular lane
	lane_type - integer discribing which lane we are talking about
		0 - left ego lane
		1 - right ego lane
		2 - far left lane
		3 - far right lane
		4 - center line
	"""
	x, y = get_x_y_values(lane_points)
	lane_poly = np.polyfit(x, y, 2)

	if(lane_type == 0):
		left_lane_polynomial_arr.append(lane_poly)
		if len(left_lane_polynomial_arr) > moving_average: left_lane_polynomial_arr.pop(0)
		return calculate_polynomial_moving_average(left_lane_polynomial_arr, moving_average)
	elif (lane_type == 1):
		right_lane_polynomial_arr.append(lane_poly)
		if len(right_lane_polynomial_arr) > moving_average: right_lane_polynomial_arr.pop(0)
		return calculate_polynomial_moving_average(right_lane_polynomial_arr, moving_average)
	elif (lane_type == 5):
		center_lane_polynomial_arr.append(lane_poly)
		if len(center_lane_polynomial_arr) > moving_average: center_lane_polynomial_arr.pop(0)
		return calculate_polynomial_moving_average(center_lane_polynomial_arr, moving_average)
	else:
		raise Exception("Lane type not implemented / recognized")


def detectLaneDeparture(lane_points, lane_polynomial, lane_type, control_line) -> bool:
	"""
	lane_points - points correspondig to a particular lane
	lane_type - integer discribing which lane we are talking about
		0 - left ego lane
		1 - right ego lane
		2 - far left lane
		3 - far right lane
	"""
	x,_ = get_x_y_values(lane_points)
	check_y = np.polyval(lane_polynomial, x)
	points_to_draw = (np.asarray([x, check_y]).T).astype(np.int32)
	lane_line = LineString(points_to_draw)

	if(lane_type == 0):
		left_control_lane = LineString(control_line)
		return lane_line.intersects(left_control_lane)
	elif(lane_type == 1):
		right_control_ane = LineString(control_line)
		return lane_line.intersects(right_control_ane)
	else:
		raise Exception("Lane type not implemented / recognized")

def center_of_lane(left_lane, right_lane):
    # Find the midpoint of the left lane
    left_x = sum([point[0] for point in left_lane]) / len(left_lane)
    left_y = sum([point[1] for point in left_lane]) / len(left_lane)
    left_midpoint = [left_x, left_y]
    
    # Find the midpoint of the right lane
    right_x = sum([point[0] for point in right_lane]) / len(right_lane)
    right_y = sum([point[1] for point in right_lane]) / len(right_lane)
    right_midpoint = [right_x, right_y]
    
    # Calculate the center of the lane
    center_x = (left_x + right_x) / 2
    center_y = (left_y + right_y) / 2
    center = [center_x, center_y]
    
    return center


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

		dataFile = open('trainingCorners.csv', 'w', newline='')
		writer = csv.writer(dataFile)
		writer.writerow(["state", "distance_to_left", "distance_to_right", "delta_left_lane", "delta_right_lane"])

	@staticmethod
	def initialize_model(model_path, cfg, use_gpu):

		# Load the model architecture
		net = parsingNet(pretrained = False, backbone='18', cls_dim = (cfg.griding_num+1, cfg.cls_num_per_lane, 4),	use_aux=False)

		# Load the weights from the downloaded model
		if torch.cuda.is_available():
			print("CUDA is availible - running on cuda")
			net = net.cuda()
			state_dict = torch.load(model_path, map_location='cuda')['model'] # CUDA
		else:
			print("CUDA NOT AVAILABLE - RUNNING ON DEVICE!")
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
			transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
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

		probability = scipy.special.softmax(processed_output[:-1, :, :], axis=0)
		idx = np.arange(cfg.griding_num) + 1
		idx = idx.reshape(-1, 1, 1)
		loc = np.sum(probability * idx, axis=0)
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
							int(cfg.img_h * (cfg.row_anchor[cfg.cls_num_per_lane-1-point_num]/288)) - 1 ]
						lane_points.append(lane_point)
			else:
				lanes_detected.append(False)

			lanes_points.append(lane_points)
		return np.array(lanes_points, dtype=object), np.array(lanes_detected, dtype=object)

	@staticmethod
	def draw_lanes(input_img, lanes_points, lanes_detected, cfg, draw_points=True):
		global right_lane_freeze
		global left_lane_freeze

		# Write the detected line points into the binary image
		binary_img = np.zeros((cfg.img_h, cfg.img_w, 3), dtype="uint8")
		if(lanes_detected[1] and lanes_detected[2]):

			lanes_points[1] = filter_roi_points(lanes_points[1], ROI_VERTICIES)
			lanes_points[2] = filter_roi_points(lanes_points[2], ROI_VERTICIES)

			if(len(lanes_points) == 0 or len(lanes_points[2]) == 0):
				print("DUPA")
				return input_img
			
			# draw the points
			for lane_num, lane_points in enumerate(lanes_points):
				for lane_point in lane_points:
					cv2.circle(binary_img, (int(lane_point[0]), int(lane_point[1])), 3, lane_colors[lane_num], -1)

			# calculate the lines
			try:
				left_lane_poly =  getAverageLanePolynomial(lanes_points[1], 0)
				right_lane_poly = getAverageLanePolynomial(lanes_points[2], 1)
			except:
				draw_line_from_poly(lanes_points[1], left_lane_polynomial_arr[-1], binary_img, (0, 0, 255))
				draw_line_from_poly(lanes_points[2], right_lane_polynomial_arr[-1], binary_img, (0, 255, 0))
				input_img = cv2.addWeighted(input_img, 0.5, binary_img, 0.5, 0)
				return input_img

			# draw lines
			draw_line_from_poly(lanes_points[1], left_lane_poly, binary_img, (0, 0, 255))
			draw_line_from_poly(lanes_points[2], right_lane_poly, binary_img, (0, 255, 0))

			# # calculate the center of the road lane
			# center_points = lanes_points[1] + lanes_points[2]
			# avg_center = [[x / 2 for x in sublist] for sublist in center_points]
			# for lane_point in avg_center:
			# 		cv2.circle(binary_img, (lane_point[0], lane_point[1]), 3, (155, 100, 255), 3)

			# Calculate the midpoint of the left lane
			midpoints = []
			length = min(len(lanes_points[1]), len(lanes_points[2]))
			for i in range(length):
				left_x = lanes_points[1][i][0]
				left_y = lanes_points[1][i][1]
				right_x = lanes_points[2][i][0]
				right_y = lanes_points[2][i][1]
				midpoint = [(left_x + right_x) / 2, (left_y + right_y) / 2]
				midpoints.append(midpoint)

			
			# DRAW CENTER LINE
			# calculate the lines
			
			ACCEPTED_ZONE_MARGIN = 30
			WARNING_ZONE_MARGIN = 60
			DANGER_ZONE_MARGIN = 100

			midpoints = filter_roi_points(midpoints, ROI_VERTICIES)
				# # Draw the center line
				# for x,y in midpoints:
				# 	x = int(x)
				# 	y = int(y)
				# 	cv2.circle(binary_img, (x, y), 3, (255, 0, 255), 3)
				# 	cv2.circle(binary_img, (x+ACCEPTED_ZONE_MARGIN, y), 1, (0, 255, 0), 3)
				# 	cv2.circle(binary_img, (x-ACCEPTED_ZONE_MARGIN, y), 1, (0, 255, 0), 3)

				# 	cv2.circle(binary_img, (x+WARNING_ZONE_MARGIN, y), 1, (0, 255, 255), 3)
				# 	cv2.circle(binary_img, (x-WARNING_ZONE_MARGIN, y), 1, (0, 255, 255), 3)

				# 	cv2.circle(binary_img, (x+DANGER_ZONE_MARGIN, y), 1, (0, 0, 255), 3)
				# 	cv2.circle(binary_img, (x-DANGER_ZONE_MARGIN, y), 1, (0, 0, 255), 3)

			# WARNING DEPARTURE LOGIC
			car_center = (670, 480)

			# Define the polynomial function
			f = lambda x, a, b, c : a*x**2 + b*x + c

			midpoint_x, midpoint_y = get_x_y_values(midpoints)
			midpoint_poly = np.polyfit(midpoint_x, midpoint_y, 2)
			x_vals = np.linspace(midpoint_x[0], midpoint_x[-1], num=100)
			closest_x = x_vals[np.argmin((f(x_vals, *midpoint_poly) - car_center[1])**2)]
			closest_y = f(closest_x, *midpoint_poly)
			distance_to_center = np.sqrt((car_center[0] - closest_x)**2 + (car_center[1] - closest_y)**2)

			state = 0
			if(distance_to_center < ACCEPTED_ZONE_MARGIN):
				cv2.circle(binary_img, car_center, 5, (0, 255, 0), 3)
				state = 0
			elif distance_to_center > ACCEPTED_ZONE_MARGIN and distance_to_center < WARNING_ZONE_MARGIN:
				cv2.circle(binary_img, car_center, 5, (0, 255, 255), 3)
				state = 1
			else:
				cv2.circle(binary_img, car_center, 5, (0, 0, 255), 3)
				state = 2
			
			# CALCULATE DISTANCE TO LEFT AND RIGHT LANES
			if(lanes_detected[1] and lanes_detected[2]):
				lanes_points[1] = filter_roi_points(lanes_points[1], ROI_VERTICIES)
				lanes_points[2] = filter_roi_points(lanes_points[2], ROI_VERTICIES)

				#calculate distance to the left lane
				left_poly = left_lane_polynomial_arr[-1]
				x_vals = np.linspace(lanes_points[1][0][0], lanes_points[1][-1][0], num=100)
				closest_x = x_vals[np.argmin((f(x_vals, *left_poly) - car_center[1])**2)]
				closest_y = f(closest_x, *left_poly)
				distance_to_left_temp = np.sqrt((car_center[0] - closest_x)**2 + (car_center[1] - closest_y)**2)
				#smoothen
				distance_to_left_arr.append(distance_to_left_temp)
				if len(distance_to_left_arr) > 4: distance_to_left_arr.pop(0)
				distance_to_left = sum(distance_to_left_arr) / len(distance_to_left_arr)
				# print(distance_to_left)

				#calculate distance to the right lane
				right_poly = right_lane_polynomial_arr[-1]
				x_vals = np.linspace(lanes_points[2][0][0], lanes_points[2][-1][0], num=100)
				closest_x = x_vals[np.argmin((f(x_vals, *right_poly) - car_center[1])**2)]
				closest_y = f(closest_x, *right_poly)
				distance_to_right_temp = np.sqrt((car_center[0] - closest_x)**2 + (car_center[1] - closest_y)**2)
				#smoothen
				distance_to_right_arr.append(distance_to_right_temp)
				if len(distance_to_right_arr) > 4: distance_to_right_arr.pop(0)
				distance_to_right = sum(distance_to_right_arr) / len(distance_to_right_arr)
				# print(distance_to_right)

				# calculate rate of change
				DELTA_MOVING_AVERAGE = 10
				delta_left_lane = distance_to_left_arr[-1] - distance_to_left_arr[0]
				delta_left_lane_arr.append(delta_left_lane)
				if len(delta_left_lane_arr) > DELTA_MOVING_AVERAGE: delta_left_lane_arr.pop(0)
				delta_left_lane = sum(delta_left_lane_arr) / len(delta_left_lane_arr)

				# SMOOTHEN
				delta_right_lane = distance_to_right_arr[-1] - distance_to_right_arr[0]
				delta_right_lane_arr.append(delta_right_lane)
				if len(delta_right_lane_arr) > DELTA_MOVING_AVERAGE: delta_right_lane_arr.pop(0)
				delta_right_lane = sum(delta_right_lane_arr) / len(delta_right_lane_arr)
				# print("L: ", delta_left_lane, "\tR: ", delta_right_lane)

				# open the file in the write mode
				dataFile = open('trainingCorners.csv', 'a', newline='')
				# create the csv writer
				writer = csv.writer(dataFile)
				writer.writerow([state, distance_to_left, distance_to_right, delta_left_lane, delta_right_lane]) ## add time in state
				dataFile.close()


			# ADD TIME IN ZONE COUNTER MAYBE :D

			# ############################## ROI VISUALIZED ################################
			# # Define the vertices of the trapezoid ROI
			# vertices = np.array([[(200, 600),         		# bottom-left corner
			# 					(500, 400),                	# top-left corner
			# 					(700, 400),                	# top-right corner
			# 					(1200, 600)]], dtype=np.int32) # bottom-right corner
			# # Create a mask with the same shape as the image
			# mask = np.zeros_like(input_img)
			# # Fill the polygon defined by the vertices with white color (255)
			# cv2.fillPoly(mask, vertices, 255)
			# # Apply the mask to the image
			# masked_image = cv2.bitwise_and(input_img, mask)
			# ############################## ROI VISUALIZED ################################

			# # Define the vertices of the trapezoid ROI
			# vertices = np.array([[(200, 600),         		# bottom-left corner
			# 					(500, 400),                	# top-left corner
			# 					(700, 400),                	# top-right corner
			# 					(1200, 600)]], dtype=np.int32) # bottom-right corner
			# # Create a mask with the same shape as the image
			# mask = np.zeros_like(input_img)
			# # Fill the polygon defined by the vertices with white color (255)
			# cv2.fillPoly(mask, vertices, 255)
			# # Apply the mask to the image
			# masked_image = cv2.bitwise_and(input_img, mask)
			# return masked_image


			
			input_img = cv2.addWeighted(input_img, 0.5, binary_img, 0.5, 0)

			# left_x = sum([point[0] for point in lanes_points[1]]) / len(lanes_points[1])
			# left_y = sum([point[1] for point in lanes_points[1]]) / len(lanes_points[1])
			# left_midpoint = [left_x, left_y]

			# # Calculate the midpoint of the right lane
			# right_x = sum([point[0] for point in lanes_points[2]]) / len(lanes_points[2])
			# right_y = sum([point[1] for point in lanes_points[2]]) / len(lanes_points[2])
			# right_midpoint = [right_x, right_y]

			# # Combine the midpoints into a single list of points
			# center_points = [left_midpoint, right_midpoint]

			# # Fit a 2nd degree polynomial to the x and y values of the center points
			# x_values = [point[0] for point in center_points]
			# y_values = [point[1] for point in center_points]
			# coefs = np.polyfit(y_values, x_values, 2)

			# # Define the polynomial function
			# poly_func = np.poly1d(coefs)

			# # Define the range of y values to use for the center line
			# y_range = np.linspace(0, binary_img.shape[0], 100)



			# draw_line_from_poly(lanes_points[2], center_lane_poly, binary_img, (255, 0, 0))
		


			# for x,y in zip(center_x, center_y):
			# 	cv2.circle(binary_img, (int(x), int(y)), 3, (255, 0, 255), 3)




		return input_img





		'''Below you can find my first approach by using normal view'''
		# # Draw controll lines for crossing lane logic
		# left_control_line = [(570, 460), (570, visualization_img.shape[0])]
		# right_control_line = [(680, 460), (680, visualization_img.shape[0])]
		# cv2.circle(visualization_img, (570, 460), 10, (255,192,203) , -1)
		# cv2.line(visualization_img, left_control_line[0], left_control_line[1], (255,192,203), 3) 
		# cv2.circle(visualization_img, (680, 460), 10, (0,192,0) , -1)
		# cv2.line(visualization_img, right_control_line[0], right_control_line[1], (0,192,0), 3)

		# #LEFT LANE DETECTION
		# if(lanes_detected[1]):
		# 	final_poly = getAverageLanePolynomial(lanes_points[1], 0)

		# 	if(left_lane_freeze > 0):
		# 		left_lane_freeze -= 1
		# 		final_poly = calculate_polynomial_moving_average(left_lane_polynomial_arr)
		# 		draw_line_from_poly(lanes_points[1], final_poly, visualization_img, (0, 0, 255))
		# 	else:
		# 		final_poly = calculate_polynomial_moving_average(left_lane_polynomial_arr)
		# 		draw_line_from_poly(lanes_points[1], final_poly, visualization_img, (00, 255, 00))

		# 	if(detectLaneDeparture(lanes_points[1], final_poly, 1, left_control_line)):
		# 		isLeftLaneCrossed = True
		# 		left_lane_freeze = LANE_CROSSING_FREEZER
		
		# #RIGHT LANE DETECTION
		# if(lanes_detected[2]):
		# 	final_poly = getAverageLanePolynomial(lanes_points[2], 1)

		# 	if(right_lane_freeze > 0):
		# 		right_lane_freeze -= 1
		# 		final_poly = calculate_polynomial_moving_average(right_lane_polynomial_arr)
		# 		draw_line_from_poly(lanes_points[2], final_poly, visualization_img, (0, 0, 255))
		# 	else:
		# 		final_poly = calculate_polynomial_moving_average(right_lane_polynomial_arr)
		# 		draw_line_from_poly(lanes_points[2], final_poly, visualization_img, (00, 255, 00))

		# 	if(detectLaneDeparture(lanes_points[2], final_poly, 1, right_control_line)):
		# 		isRightLaneCrossed = True
		# 		right_lane_freeze = LANE_CROSSING_FREEZER

		# return visualization_img
