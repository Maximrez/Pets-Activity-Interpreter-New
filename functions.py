from PIL import ImageOps
import os
import cv2
import shutil
import time
import numpy as np
from PIL import Image
from catboost import CatBoostClassifier
from torchvision import transforms
from cv2 import FONT_HERSHEY_PLAIN
from settings import *
from models import *
from converter import Converter
import matplotlib.pyplot as plt


def max_index(values):
    m_value = 0
    m_idx = 0
    for idx in range(len(values)):
        if values[idx] > m_value:
            m_value = values[idx]
            m_idx = idx
    return m_idx, m_value


def image_padding(image, target_padding, target_size):
    w_size, h_size = image.size

    x_padding = int(target_padding[0] * w_size / (target_size[0] - 2 * target_padding[0]))
    y_padding = int(target_padding[1] * h_size / (target_size[1] - 2 * target_padding[1]))

    image = ImageOps.expand(image, border=(x_padding, y_padding), fill='white')
    return image, x_padding, y_padding


def rescale(x, y, crop_x1, crop_x2, crop_y1, crop_y2, x_padding, y_padding, target_size):
    x = int(x * (crop_x2 - crop_x1 + 2 * x_padding) / target_size[0]) - x_padding + crop_x1
    y = int(y * (crop_y2 - crop_y1 + 2 * y_padding) / target_size[1]) - y_padding + crop_y1
    return x, y


def process_video(file_path, project_dir, out_path, show_window=False, show_scores=False):
    torch.manual_seed(22)

    data_dir = os.path.join(project_dir, 'data')
    yolo_dir = os.path.join(data_dir, 'yolov3')

    shutil.unpack_archive(os.path.join(data_dir, 'yolov3.zip'), yolo_dir)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    net = cv2.dnn.readNet(os.path.join(yolo_dir, "yolov3.weights"), os.path.join(yolo_dir, "yolov3.cfg"))

    layer_names = net.getLayerNames()
    outputlayers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    with open(os.path.join(yolo_dir, "coco.names"), "r") as f:
        classes = [line.strip() for line in f.readlines()]

    model_keypoints = keypointrcnn_mobilenet('mobilenet_v3_large',
                                             os.path.join(data_dir, 'models', 'keypointrcnn_mobilenetv3large.pth'),
                                             device)

    test_transforms = transforms.Compose([transforms.Resize(target_size), transforms.ToTensor()])

    cat_classifier = CatBoostClassifier()
    cat_classifier.load_model(os.path.join(data_dir, 'models', 'cat_classifier_augm6'))

    cat_classifier1 = CatBoostClassifier()
    cat_classifier1.load_model(os.path.join(data_dir, 'models', 'cat_classifier_last4'))

    cap = cv2.VideoCapture(file_path)
    cap_fps = int(cap.get(cv2.CAP_PROP_FPS))

    _, frame = cap.read()
    frame_height, frame_width, frame_channels = frame.shape

    if show_window:
        if frame_width > max_picture_shape[0]:
            frame_height = int(frame_height * max_picture_shape[0] / frame_width)
            frame_width = max_picture_shape[0]
        if frame_height > max_picture_shape[1]:
            frame_width = int(frame_width * max_picture_shape[1] / frame_height)
            frame_height = max_picture_shape[1]

    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    cap_out = cv2.VideoWriter(out_path, fourcc, cap_fps, (frame_width, frame_height))

    min_score = MIN_SCORE
    if show_scores:
        min_score = -9999

    prev_keys_position = [(-1, -1)] * 20
    sum_movement = 0
    activity_count = [0] * len(activity_classes)
    stats_m = []
    stats_a = []

    starting_time = time.time()
    frame_id = 0

    while True:
        _, frame = cap.read()
        if frame is None:
            break
        frame_id += 1

        frame = cv2.resize(frame, (frame_width, frame_height))

        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), bgr_colors['k'], True, crop=False)

        if (frame_id - 1) % (cap_fps // NUMBER_PROCESSED_FRAMES) == 0:
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
                        center_x = int(detection[0] * frame_width)
                        center_y = int(detection[1] * frame_height)
                        w = int(detection[2] * frame_width)
                        h = int(detection[3] * frame_height)
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
                    crop_x2 = min(x + w, frame_width)
                    crop_y2 = min(y + h, frame_height)

                    cv2.rectangle(frame, (crop_x1, crop_y1), (crop_x2, crop_y2), bgr_colors['g'], 2)

                    cv2.putText(frame, label, (crop_x1 + 10, crop_y1 + 20), FONT_HERSHEY_PLAIN, 1.5, bgr_colors['w'], 2)
                    cv2.putText(frame, str(round(confidence, 2)), (crop_x1 + 90, crop_y1 + 20), FONT_HERSHEY_PLAIN, 1,
                                bgr_colors['w'], 2)

                    crop_img = frame[crop_y1:crop_y2, crop_x1:crop_x2, :]
                    im_pil = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))

                    im_pil, x_padding, y_padding = image_padding(im_pil, TARGET_PADDING, target_size)
                    inputs = test_transforms(im_pil).unsqueeze(0).to(device)

                    if (frame_id - 1) % (cap_fps // NUMBER_PROCESSED_FRAMES) == 0:
                        with torch.no_grad():
                            outputs = model_keypoints(inputs)[0]

                            for key in outputs.keys():
                                outputs[key] = outputs[key].cpu()

                            keypoints = (outputs['keypoints'][0].detach().numpy())[:, :].reshape(-1, 3)
                            keypoints_scores = outputs['keypoints_scores'][0].numpy()
                            bbox = outputs['boxes'][0].numpy()

                    for k in range(keypoints.shape[0]):
                        if keypoints[k, 2] == 1 and keypoints_scores[k] > min_score:
                            keypoints[k, 2] = 1
                        else:
                            keypoints[k] = [0., 0., 0.]

                    for l in lines:
                        if keypoints[l[0], 2] == 1 and keypoints[l[1], 2] == 1:
                            x1, y1 = rescale(keypoints[l[0], 0], keypoints[l[0], 1], crop_x1, crop_x2, crop_y1, crop_y2,
                                             x_padding, y_padding, target_size)
                            x2, y2 = rescale(keypoints[l[1], 0], keypoints[l[1], 1], crop_x1, crop_x2, crop_y1, crop_y2,
                                             x_padding, y_padding, target_size)

                            cv2.line(frame, (x1, y1), (x2, y2), l[2], 2)

                    list_features = list(bbox)

                    for k in range(keypoints.shape[0]):
                        keypoint = keypoints[k]
                        list_features.append(keypoint[2])
                        list_features.append(keypoint[0])
                        list_features.append(keypoint[1])

                        if keypoint[2] == 1:
                            x0, y0 = rescale(keypoint[0], keypoint[1],
                                             crop_x1, crop_x2, crop_y1, crop_y2, x_padding, y_padding, target_size)

                            cv2.circle(frame, (x0, y0), radius=2, color=(0, 0, 0), thickness=2)
                            if show_scores:
                                cv2.putText(frame, str(round(keypoints_scores[k], 2)), (x0 + 5, y0 + 5),
                                            FONT_HERSHEY_PLAIN, 0.8, bgr_colors['w'], 1)

                            if prev_keys_position[k] != (-1, -1):
                                sum_movement += ((prev_keys_position[k][0] - x0) ** 2 + (
                                        prev_keys_position[k][1] - y0) ** 2) ** 0.5
                            prev_keys_position[k] = (x0, y0)

                    list_features.append(animal_classes.index(label))

                    cat_preds = cat_classifier.predict(list_features, prediction_type='Probability')
                    if len(cat_preds) == 2:
                        cat_preds1 = cat_classifier1.predict(list_features, prediction_type='Probability')
                        cat_preds = np.concatenate(([cat_preds[0]], cat_preds1 * cat_preds[1]))

                    max_idx, max_prob = max_index(cat_preds)
                    activity = activity_classes[max_idx]
                    cv2.putText(frame, activity, (crop_x1 + 10, crop_y1 + 45), FONT_HERSHEY_PLAIN, 1.5, bgr_colors['w'],
                                2)
                    cv2.putText(frame, str(round(max_prob, 2)), (crop_x1 + 90, crop_y1 + 45), FONT_HERSHEY_PLAIN, 1,
                                bgr_colors['w'], 2)

                    activity_count[max_idx] += 1
        if (frame_id + 1) % cap_fps == 0:
            stats_m.append(sum_movement / cap_fps)
            max_activity_idx, _ = max_index(activity_count)
            stats_a.append(activity_classes[max_activity_idx])

            prev_keys_position = [(-1, -1)] * 20
            sum_movement = 0
            activity_count = [0] * len(activity_classes)

        elapsed_time = time.time() - starting_time
        fps = frame_id / elapsed_time

        if show_window:
            cv2.putText(frame, "FPS: " + str(round(fps, 2)), (5, 15), FONT_HERSHEY_PLAIN, 1, bgr_colors['w'], 2)
            cv2.imshow('Video', frame)
            key = cv2.waitKey(1)
            if key == 27:
                break

        cap_out.write(frame)

    elapsed_time = time.time() - starting_time
    fps = frame_id / elapsed_time
    spf = elapsed_time / frame_id
    print(f"seconds per frame: {spf}, frames per second: {fps}")

    cap.release()
    cap_out.release()
    cv2.destroyAllWindows()

    return stats_m, stats_a
    # return frame_width, frame_height, cap_fps


def convert_avi_to_mp4(avi_path, mp4_path, ff_bin_path, width, height, fps):
    ffmpeg_path = os.path.join(ff_bin_path, 'ffmpeg.exe')
    ffprobe_path = os.path.join(ff_bin_path, 'ffprobe.exe')

    conv = Converter(ffmpeg_path, ffprobe_path)

    options = {'format': 'mp4',
               'audio': {
                   'codec': 'aac',
                   'samplerate': 11025,
                   'channels': 2
               },
               'video': {
                   'codec': 'hevc',
                   'width': width,
                   'height': height,
                   'fps': fps
               }}

    convert = conv.convert(avi_path, mp4_path, options)
    try:
        for timecode in convert:
            print(f'\rConverting ({timecode:.2f}) ...')
    except AttributeError:
        pass


def stats_graph(stats_m, stats_a, graph_path, show_graph=False):
    plt.rcParams['figure.figsize'] = (1 + len(stats_m), 5)

    bar_colors = list(activity_colors[stat] for stat in stats_a)
    plt.bar(range(1, len(stats_m) + 1), stats_m, color=bar_colors)
    plt.ylabel('Activity')
    plt.xlabel('Seconds')
    plt.title("Activity histogram")

    if show_graph:
        plt.show()
    plt.savefig(graph_path)

    plt.gcf().clear()
