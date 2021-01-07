import os
import numpy as np
from scipy import spatial
import time
import cv2
import csv

from .gui import show_error_and_quit

class VideoProcesser:
    def __init__(self, ui_handle=None):
        self.ui_handle = ui_handle
    
    def set_model(self, yolo_model):
        self.yolo_model = yolo_model

    def open_io(self, input_path, output_path):
        self.video_in = cv2.VideoCapture(input_path)
        if not self.video_in.isOpened():
            show_error_and_quit('Corrupt video file.', 'Program will now exit.')

        vid_width = int(self.video_in.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid_height = int(self.video_in.get(cv2.CAP_PROP_FRAME_HEIGHT))
        vid_fps = self.video_in.get(cv2.CAP_PROP_FPS)
        vid_ext = input_path.split(".")[-1]

        if (vid_ext == "avi"):
            self.video_out = cv2.VideoWriter(
                output_path,
                cv2.VideoWriter_fourcc(*"MJPG"),
                vid_fps,
                (vid_width, vid_height)
                )
        elif (vid_ext == "mp4"):
            self.video_out = cv2.VideoWriter(
                output_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                vid_fps,
                (vid_width, vid_height)
                )
        else:
            # i dunno, do something wise
            self.video_out = cv2.VideoWriter(
                output_path,
                cv2.VideoWriter_fourcc(*"MJPG"),
                vid_fps,
                (vid_width, vid_height)
                )

        f = open(output_path + '.csv', 'w', newline='')
        self.csv_out = csv.DictWriter(f, fieldnames=['id', 'label', 'confidence', 'appears_at', 'disappears_at'])

    def run(self):
        fps = 0
        num_frames = 0
        try:
            num_frames = int(self.video_in.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = self.video_in.get(cv2.CAP_PROP_FPS)
        except:
            show_error_and_quit('Cant read properties from file', 'Program will now exit.')

        if fps <= 0 or num_frames < 1:
            show_error_and_quit('Cant read properties from file', 'Program will now exit.')
        
        start_time = time.time()
        last_time = start_time
        
        # tracking previous frames, in the beginning empty
        previous_frame_detections = {}

        # vehicle list
        vehicles = []

        frame_counter = 1
        while True:
            current_seconds = (frame_counter - 1) / fps
            
            success, frame = self.video_in.read()

            if not success:
                break

            boxes, confidences, classIDs = extract_vehicles_from_frame(frame, self.yolo_model)

            (net, LABELS, COLORS) = self.yolo_model
            
            new_frame_objects = {}
            tracked_objects = {}

            # track objects from previous frames on current frame
            for obj in previous_frame_detections.values():
                ok, bbox = obj['tracker'].update(frame)
                if ok:
                    (x, y, w, h) = bbox
                    x = int(x)
                    y = int(y)
                    w = int(w)
                    h = int(h)
                    centerX = x + (w//2)
                    centerY = y + (h//2)
                    bbox = (x, y, w, h)

                    tracked_objects[(centerX, centerY)] = obj

                    # draw a bounding box rectangle and label on the image
                    vehicle_data = vehicles[obj['id']]
                    color = COLORS[vehicle_data['label']]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.rectangle(frame, (x, y - 20), (x + w, y), color, -1) # for rectangle below text
                    text = "{} {}: {:.4f}".format(obj['id'], vehicle_data['label'], vehicle_data['confidence'])
                    cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, [255, 255, 255], 2)

                    vehicles[obj['id']]['disappears_at'] = current_seconds

            # apply non-maxima suppression to suppress weak, overlapping bounding
            # boxes
            idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.7)

            fr_width  = self.video_in.get(cv2.CAP_PROP_FRAME_WIDTH)
            fr_height = self.video_in.get(cv2.CAP_PROP_FRAME_HEIGHT)

            # ensure at least one detection exists
            if len(idxs) > 0:
            	# loop over the indexes we are keeping
            	for i in idxs.flatten():
                    if omit_match(LABELS[classIDs[i]]):
                        continue

                    actual_label = change_label(LABELS[classIDs[i]])

            		# extract the bounding box coordinates
                    (x, y) = (boxes[i][0], boxes[i][1])
                    (w, h) = (boxes[i][2], boxes[i][3])
                    centerX = x + (w//2)
                    centerY = y + (h//2)

                    if w < 100 or h < 100:
                        continue

                    ignore_threshold = 20 # px
                    if x < ignore_threshold or x > fr_width - ignore_threshold:
                        continue

                    if y < ignore_threshold or y > fr_height - ignore_threshold:
                        continue
                    
                    coordinate_list = list(tracked_objects.keys())

                    if len(coordinate_list) > 0:
                        dist = np.Inf
                        # Finding the distance to the closest point and the index
                        temp_dist, index = spatial.KDTree(coordinate_list).query([(centerX, centerY)])
                        if (temp_dist < dist):
                            dist = temp_dist

                        if (dist < (max(w, h)) / 2):
    	                    continue

                    tracker = cv2.TrackerKCF_create()
                    tracker.init(frame, (x, y, w, h))
                    new_object = {
                        'id': len(vehicles),
                        'classID': classIDs[i],
                        'confidence': confidences[i],
                        'tracker': tracker
                        }

                    vehicles.append({
                        'label': actual_label,
                        'confidence': confidences[i],
                        'appears_at': current_seconds,
                        'disappears_at': current_seconds
                    })

                    new_frame_objects[(centerX, centerY)] = new_object

                    # draw a bounding box rectangle and label on the image
                    vehicle_data = vehicles[new_object['id']]
                    color = COLORS[vehicle_data['label']]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.rectangle(frame, (x, y - 20), (x + w, y), color, -1) # for rectangle below text
                    text = "{} {}: {:.4f}".format(new_object['id'], vehicle_data['label'], vehicle_data['confidence'])
                    cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, [255, 255, 255], 2)

            previous_frame_detections = new_frame_objects
            previous_frame_detections.update(tracked_objects)

            self.video_out.write(frame)
            frame_counter = frame_counter + 1

            # measure fps
            current_time = time.time()
            avg_fps = frame_counter / (current_time - start_time)
            current_fps = 1 / (current_time - last_time)

            self.ui_handle.update_status('Frame ' + str(frame_counter) + '/' + str(num_frames) + '  (avg ' + '{:.2f}'.format(avg_fps) + ' current ' + '{:.2f}'.format(current_fps) + '  FPS)')

            last_time = current_time

        self.video_in.release()
        self.video_out.release()

        self.csv_out.writeheader()
        for i in range(0, len(vehicles)):
            veh = vehicles[i]
            veh['id'] = str(i)
            self.csv_out.writerow(veh)

def load_model():
    MODEL_DIR_NAME = 'model_data'

    # load the COCO class labels our YOLO model was trained on
    labels_path = os.path.sep.join([MODEL_DIR_NAME, "coco.names"])
    LABELS = open(labels_path).read().strip().split("\n")

    weights_path = os.path.sep.join([MODEL_DIR_NAME, "yolo.weights"])
    config_path = os.path.sep.join([MODEL_DIR_NAME, "yolo.cfg"])

    net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
    
    COLORS = {
        'car': [255, 0, 0],
        'motorbike': [0, 255, 0],
        'truck/bus': [0, 0, 255],
        'other': [127, 127, 0]
    }
    
    return (net, LABELS, COLORS)

def change_label(classID):
      # Concatenate yolos classes in my one, bigger class
    classID_map = {
        'car': 'car',
        'motorbike': 'motorbike',
        'bicycle': 'motorbike',
        'truck': 'truck/bus',
        'bus': 'truck/bus'
    }
	
    # Check if there is necessity to override classID
    if classID in classID_map.keys():
        return classID_map[str(classID)]
    else:
        return "other"

def omit_match(classID):
    # Array of unneccesary matches
    not_match = ["traffic light"]

    # Omit unnecessary matches
    if classID in not_match:
        return True
    else:
        return False

def box_in_previous_frames(previous_frame_detections, current_box, frames_before_current_val):
    centerX, centerY, width, height = current_box
    dist = np.inf #Initializing the minimum distance
    # Iterating through all the k-dimensional trees
    for i in range(frames_before_current_val):
        coordinate_list = list(previous_frame_detections[i].keys())
        if len(coordinate_list) == 0: # When there are no detections in the previous frame
            continue
        # Finding the distance to the closest point and the index
        temp_dist, index = spatial.KDTree(coordinate_list).query([(centerX, centerY)])
        if (temp_dist < dist):
            dist = temp_dist
            frame_num = i
            coord = coordinate_list[index[0]]

    if (dist > (max(width, height)) / 2):
    	return False, None

    # Keeping the vehicle ID constant
    return True, previous_frame_detections[frame_num][coord]

def extract_vehicles_from_frame(image, yolo_model):
    (net, LABELS, COLORS) = yolo_model
    (H, W) = image.shape[:2]
    # determine only the *output* layer names that we need from YOLO
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    # construct a blob from the input image and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes and
    # associated probabilities
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
    	swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)
    # show timing information on YOLO
    # print("[INFO] YOLO took {:.6f} seconds".format(end - start))
    # initialize our lists of detected bounding boxes, confidences, and
    # class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []
    # loop over each of the layer output
    for output in layerOutputs:
    	# loop over each of the detections
    	for detection in output:
    		# extract the class ID and confidence (i.e., probability) of
    		# the current object detection
    		scores = detection[5:]
    		classID = np.argmax(scores)
    		confidence = scores[classID]
    		# filter out weak predictions by ensuring the detected
    		# probability is greater than the minimum probability
    		if confidence > 0.5:
    			# scale the bounding box coordinates back relative to the
    			# size of the image, keeping in mind that YOLO actually
    			# returns the center (x, y)-coordinates of the bounding
    			# box followed by the boxes' width and height
    			box = detection[0:4] * np.array([W, H, W, H])
    			(centerX, centerY, width, height) = box.astype("int")
    			# use the center (x, y)-coordinates to derive the top and
    			# and left corner of the bounding box
    			x = int(centerX - (width / 2))
    			y = int(centerY - (height / 2))
    			# update our list of bounding box coordinates, confidences,
    			# and class IDs
    			boxes.append([x, y, int(width), int(height)])
    			confidences.append(float(confidence))
    			classIDs.append(classID)

    return boxes, confidences, classIDs
