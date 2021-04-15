import os
import cv2
import shutil
import torch
import time
import torchvision
from torchvision.models.detection import KeypointRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch.distributed as dist
import torch.nn.functional as F

torch.manual_seed(22)

target_size = (256, 256)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device:", device)

data_dir = 'D:/PycharmProjects/CV_project/data'
yolo_dir = 'D:/PycharmProjects/CV_project/data/yolov3'

shutil.unpack_archive('D:/PycharmProjects/CV_project/data/yolov3.zip', yolo_dir)

net = cv2.dnn.readNet(os.path.join(yolo_dir, "yolov3.weights"), os.path.join(yolo_dir, "yolov3.cfg"))
layer_names = net.getLayerNames()
outputlayers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

with open(os.path.join(yolo_dir, "coco.names"), "r") as f:
    classes = [line.strip() for line in f.readlines()]

backbone = torchvision.models.mobilenet_v3_large(pretrained=True).features
backbone.out_channels = 960

anchor_generator = AnchorGenerator(sizes=((16, 32, 64, 128, 256),), aspect_ratios=((0.5, 1.0, 2.0),))

roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)
keypoint_roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=14, sampling_ratio=2)
model_keypoints = KeypointRCNN(backbone, num_classes=6, num_keypoints=20, rpn_anchor_generator=anchor_generator,
                               box_roi_pool=roi_pooler, keypoint_roi_pool=keypoint_roi_pooler)
model_keypoints = model_keypoints.to(device)
model_keypoints.load_state_dict(
    torch.load(os.path.join(data_dir, 'models', 'keypointrcnn_mobilenetv3large.pth'), map_location=device))
model_keypoints.eval()

bgr_colors = {'r': (0, 0, 255),
              'g': (0, 255, 0),
              'b': (255, 0, 0),
              'c': (255, 255, 0),
              'm': (255, 0, 255),
              'y': (0, 255, 255),
              'w': (255, 255, 255),
              'k': (0, 0, 0)}

lines = [(0, 1, bgr_colors['c']),
         (0, 4, bgr_colors['c']),
         (1, 4, bgr_colors['c']),
         (0, 2, bgr_colors['y']),
         (1, 3, bgr_colors['y']),
         (4, 5, bgr_colors['b']),
         (5, 7, bgr_colors['b']),
         (5, 8, bgr_colors['m']),
         (8, 12, bgr_colors['r']),
         (12, 16, bgr_colors['g']),
         (5, 9, bgr_colors['m']),
         (9, 13, bgr_colors['r']),
         (13, 17, bgr_colors['g']),
         (7, 6, bgr_colors['b']),
         (6, 10, bgr_colors['m']),
         (10, 14, bgr_colors['r']),
         (14, 18, bgr_colors['g']),
         (6, 11, bgr_colors['m']),
         (11, 15, bgr_colors['r']),
         (15, 19, bgr_colors['g'])]

test_transforms = transforms.Compose([transforms.Resize(target_size), transforms.ToTensor()])

test_data_dir = os.path.join(data_dir, 'test')
shutil.unpack_archive(os.path.join(data_dir, 'test.zip'), data_dir)
file_name = "dog2.mp4"
cap = cv2.VideoCapture(os.path.join(test_data_dir, "videos", file_name))

font = cv2.FONT_HERSHEY_PLAIN
starting_time = time.time()
frame_id = 0

while True:
    _, frame = cap.read()
    if frame is None:
        break
    frame_id += 1

    height, width, channels = frame.shape

    max_shape = (1500, 800)
    if width > max_shape[0]:
        height = int(height * max_shape[0] / width)
        width = max_shape[0]
        frame = cv2.resize(frame, (width, height))
    if height > max_shape[1]:
        width = int(width * max_shape[1] / height)
        height = max_shape[1]
        frame = cv2.resize(frame, (width, height))

    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), bgr_colors['k'], True, crop=False)
    net.setInput(blob)
    outs = net.forward(outputlayers)

    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.6)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            x = int(x - 0.05 * w)
            y = int(y - 0.05 * h)
            w = int(w * 1.1)
            h = int(h * 1.1)

            label = str(classes[class_ids[i]])
            if label in ['cat', 'cow', 'dog', 'horse', 'sheep']:
                crop_x1 = max(x, 0)
                crop_y1 = max(y, 0)
                crop_x2 = min(x + w, width)
                crop_y2 = min(y + h, height)

                cv2.rectangle(frame, (crop_x1, crop_y1), (crop_x2, crop_y2), bgr_colors['g'], 2)
                cv2.putText(frame, label, (x + 5, y + 15), font, 1, bgr_colors['w'], 2)

                crop_img = frame[crop_y1:crop_y2, crop_x1:crop_x2, :]
                im_pil = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))

                w_size, h_size = im_pil.size
                target_padding = (30, 30)
                x_padding = int(target_padding[0] * w_size / (target_size[0] - 2 * target_padding[0]))
                y_padding = int(target_padding[1] * h_size / (target_size[1] - 2 * target_padding[1]))

                im_pil = ImageOps.expand(im_pil, border=(x_padding, y_padding), fill='white')
                image = test_transforms(im_pil).unsqueeze(0)

                inputs = image.to(device)

                with torch.no_grad():
                    outputs = model_keypoints(inputs)[0]

                    for key in outputs.keys():
                        outputs[key] = outputs[key].cpu()

                    keypoints = (outputs['keypoints'][0].detach().numpy())[:, :].reshape(-1, 3)
                    keypoints_scores = outputs['keypoints_scores'][0].numpy()
                    min_score = 1

                    for l in lines:
                        if keypoints[l[0], 2] == 1 and keypoints_scores[l[0]] > min_score and \
                                keypoints[l[1], 2] == 1 and keypoints_scores[l[1]] > min_score:
                            x1 = int(keypoints[l[0], 0] *
                                     (crop_x2 - crop_x1 + 2 * x_padding) / target_size[0]) - x_padding + crop_x1
                            y1 = int(keypoints[l[0], 1] *
                                     (crop_y2 - crop_y1 + 2 * y_padding) / target_size[1]) - y_padding + crop_y1
                            x2 = int(keypoints[l[1], 0] *
                                     (crop_x2 - crop_x1 + 2 * x_padding) / target_size[0]) - x_padding + crop_x1
                            y2 = int(keypoints[l[1], 1] *
                                     (crop_y2 - crop_y1 + 2 * y_padding) / target_size[1]) - y_padding + crop_y1

                            cv2.line(frame, (x1, y1), (x2, y2), l[2], 2)

                    for k in range(keypoints.shape[0]):
                        if keypoints[k, 2] == 1 and keypoints_scores[k] > min_score:
                            x0 = int(keypoints[k, 0] *
                                     (crop_x2 - crop_x1 + 2 * x_padding) / target_size[0]) - x_padding + crop_x1
                            y0 = int(keypoints[k, 1] *
                                     (crop_y2 - crop_y1 + 2 * y_padding) / target_size[1]) - y_padding + crop_y1
                            cv2.circle(frame, (x0, y0), radius=2, color=(0, 0, 0), thickness=2)

    elapsed_time = time.time() - starting_time
    fps = frame_id / elapsed_time
    cv2.putText(frame, "FPS: " + str(round(fps, 2)), (5, 15), font, 1, bgr_colors['w'], 2)

    cv2.imshow(file_name, frame)
    key = cv2.waitKey(1)

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
