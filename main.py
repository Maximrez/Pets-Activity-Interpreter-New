import os
import cv2
import shutil
import time
import numpy as np
from PIL import Image
from catboost import CatBoostClassifier
from torchvision import transforms
from cv2 import FONT_HERSHEY_PLAIN
from functions import *
from settings import *
from models import *

torch.manual_seed(22)

CONFIDENCE_EDGE = 0.6
TARGET_PADDING = (30, 30)
MIN_SCORE = 1
BOX_INCREASE = 0.05

FILE_NAME = "kizoa.mp4"

data_dir = 'D:/PycharmProjects/CV_project/data'
yolo_dir = 'D:/PycharmProjects/CV_project/data/yolov3'

shutil.unpack_archive('D:/PycharmProjects/CV_project/data/yolov3.zip', yolo_dir)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device:", device)

net = cv2.dnn.readNet(os.path.join(yolo_dir, "yolov3.weights"), os.path.join(yolo_dir, "yolov3.cfg"))
layer_names = net.getLayerNames()
outputlayers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

with open(os.path.join(yolo_dir, "coco.names"), "r") as f:
    classes = [line.strip() for line in f.readlines()]

model_keypoints = keypointrcnn_mobilenet('mobilenet_v3_large',
                                         os.path.join(data_dir, 'models', 'keypointrcnn_mobilenetv3large.pth'),
                                         device)

test_transforms = transforms.Compose([transforms.Resize(target_size), transforms.ToTensor()])

cat_classifier = CatBoostClassifier()
cat_classifier.load_model(os.path.join(data_dir, 'models', 'cat_classifier'))

test_data_dir = os.path.join(data_dir, 'test', 'videos')
shutil.unpack_archive(os.path.join(data_dir, 'test_videos.zip'), os.path.join(data_dir, 'test'))

cap = cv2.VideoCapture(os.path.join(test_data_dir, FILE_NAME))

starting_time = time.time()
frame_id = 0

while True:
    _, frame = cap.read()
    if frame is None:
        break
    frame_id += 1

    height, width, channels = frame.shape

    if width > max_picture_shape[0]:
        height = int(height * max_picture_shape[0] / width)
        width = max_picture_shape[0]
        frame = cv2.resize(frame, (width, height))
    if height > max_picture_shape[1]:
        width = int(width * max_picture_shape[1] / height)
        height = max_picture_shape[1]
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
            if confidence > CONFIDENCE_EDGE:
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
            confidence = confidences[i]
            x = int(x - BOX_INCREASE * w)
            y = int(y - BOX_INCREASE * h)
            w = int(w * (1 + 2 * BOX_INCREASE))
            h = int(h * (1 + 2 * BOX_INCREASE))

            label = str(classes[class_ids[i]])
            if label in animal_classes:
                crop_x1 = max(x, 0)
                crop_y1 = max(y, 0)
                crop_x2 = min(x + w, width)
                crop_y2 = min(y + h, height)

                cv2.rectangle(frame, (crop_x1, crop_y1), (crop_x2, crop_y2), bgr_colors['g'], 2)
                cv2.putText(frame, label + " " + str(round(confidence, 2)),
                            (x + 5, y + 15), FONT_HERSHEY_PLAIN, 1, bgr_colors['w'], 2)

                crop_img = frame[crop_y1:crop_y2, crop_x1:crop_x2, :]
                im_pil = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))

                im_pil, x_padding, y_padding = image_padding(im_pil, TARGET_PADDING, target_size)
                inputs = test_transforms(im_pil).unsqueeze(0).to(device)

                with torch.no_grad():
                    outputs = model_keypoints(inputs)[0]

                    for key in outputs.keys():
                        outputs[key] = outputs[key].cpu()

                    keypoints = (outputs['keypoints'][0].detach().numpy())[:, :].reshape(-1, 3)
                    keypoints_scores = outputs['keypoints_scores'][0].numpy()
                    bbox = outputs['boxes'][0].numpy()

                    for k in range(keypoints.shape[0]):
                        if keypoints[k, 2] == 1 and keypoints_scores[k] > MIN_SCORE:
                            keypoints[k, 2] = 1
                        else:
                            keypoints[k] = [0., 0., 0.]

                    for l in lines:
                        if keypoints[l[0], 2] == 1 and keypoints[l[1], 2] == 1:
                            x1, y1 = rescale(keypoints[l[0], 0], keypoints[l[0], 1],
                                             crop_x1, crop_x2, crop_y1, crop_y2, x_padding, y_padding, target_size)
                            x2, y2 = rescale(keypoints[l[1], 0], keypoints[l[1], 1],
                                             crop_x1, crop_x2, crop_y1, crop_y2, x_padding, y_padding, target_size)

                            cv2.line(frame, (x1, y1), (x2, y2), l[2], 2)

                    list_features = list(bbox)

                    for keypoint in keypoints:
                        list_features.append(keypoint[2])
                        list_features.append(keypoint[0])
                        list_features.append(keypoint[1])

                        if keypoint[2] == 1:
                            x0, y0 = rescale(keypoint[0], keypoint[1],
                                             crop_x1, crop_x2, crop_y1, crop_y2, x_padding, y_padding, target_size)

                            cv2.circle(frame, (x0, y0), radius=2, color=(0, 0, 0), thickness=2)
                            # cv2.putText(frame, str(round(keypoints_scores[k], 2)), (x0 + 5, y0 + 5), FONT_HERSHEY_PLAIN, 0.8, bgr_colors['w'], 1)

                    list_features.append(animal_classes.index(label))

                    cat_preds = cat_classifier.predict(list_features, prediction_type='Probability')
                    max_idx, max_prob = max_confidence(cat_preds)
                    cv2.putText(frame, activity_classes[max_idx] + " " + str(round(max_prob, 2)),
                                (x + 5, y + 35), FONT_HERSHEY_PLAIN, 1, bgr_colors['w'], 2)

    elapsed_time = time.time() - starting_time
    fps = frame_id / elapsed_time
    cv2.putText(frame, "FPS: " + str(round(fps, 2)), (5, 15), FONT_HERSHEY_PLAIN, 1, bgr_colors['w'], 2)

    cv2.imshow(FILE_NAME, frame)
    key = cv2.waitKey(1)

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
