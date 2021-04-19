import os
import cv2
from tqdm import tqdm
import xmltodict
import pandas as pd

answers_dir = 'data/animalpose_keypoint/answers'
images_dir = 'data/animalpose_keypoint/images'

class_name = 'sheep'

dictionary = {'id': [],
              'cat': [], 'cow': [], 'dog': [], 'horse': [], 'sheep': [],
              'bounds_x': [], 'bounds_y': [], 'bounds_width': [], 'bounds_height': [],
              'L_eye_v': [], 'L_eye_x': [], 'L_eye_y': [],
              'R_eye_v': [], 'R_eye_x': [], 'R_eye_y': [],
              'L_ear_v': [], 'L_ear_x': [], 'L_ear_y': [],
              'R_ear_v': [], 'R_ear_x': [], 'R_ear_y': [],
              'Nose_v': [], 'Nose_x': [], 'Nose_y': [],
              'Throat_v': [], 'Throat_x': [], 'Throat_y': [],
              'Tail_v': [], 'Tail_x': [], 'Tail_y': [],
              'withers_v': [], 'withers_x': [], 'withers_y': [],
              'L_F_elbow_v': [], 'L_F_elbow_x': [], 'L_F_elbow_y': [],
              'R_F_elbow_v': [], 'R_F_elbow_x': [], 'R_F_elbow_y': [],
              'L_B_elbow_v': [], 'L_B_elbow_x': [], 'L_B_elbow_y': [],
              'R_B_elbow_v': [], 'R_B_elbow_x': [], 'R_B_elbow_y': [],
              'L_F_knee_v': [], 'L_F_knee_x': [], 'L_F_knee_y': [],
              'R_F_knee_v': [], 'R_F_knee_x': [], 'R_F_knee_y': [],
              'L_B_knee_v': [], 'L_B_knee_x': [], 'L_B_knee_y': [],
              'R_B_knee_v': [], 'R_B_knee_x': [], 'R_B_knee_y': [],
              'L_F_paw_v': [], 'L_F_paw_x': [], 'L_F_paw_y': [],
              'R_F_paw_v': [], 'R_F_paw_x': [], 'R_F_paw_y': [],
              'L_B_paw_v': [], 'L_B_paw_x': [], 'L_B_paw_y': [],
              'R_B_paw_v': [], 'R_B_paw_x': [], 'R_B_paw_y': [],
              'stand': [], 'sit': [], 'lie': [], 'go': [], 'run': [], 'jump': [], 'interact': [], 'sleep': [],
              'eat': []}

index = 777


def show_image(image):
    k = -1
    while k == -1:
        cv2.imshow(object_name[:-4], image)
        k = cv2.waitKey(33)

    cv2.destroyAllWindows()
    return k


for object_name in tqdm(os.listdir(os.path.join(answers_dir, class_name))):
    with open(os.path.join(answers_dir, class_name, object_name)) as fd:
        doc = xmltodict.parse(fd.read())

        dictionary['id'].append('0' * (3 - len(str(index))) + str(index))
        dictionary['cat'].append(int(class_name == 'cat'))
        dictionary['cow'].append(int(class_name == 'cow'))
        dictionary['dog'].append(int(class_name == 'dog'))
        dictionary['horse'].append(int(class_name == 'horse'))
        dictionary['sheep'].append(int(class_name == 'sheep'))
        dictionary['bounds_x'].append(int(doc['annotation']['visible_bounds']['@xmin']))
        dictionary['bounds_y'].append(int(doc['annotation']['visible_bounds']['@xmax']))
        dictionary['bounds_width'].append(int(doc['annotation']['visible_bounds']['@width']))
        dictionary['bounds_height'].append(int(doc['annotation']['visible_bounds']['@height']))

        for keypoint in doc['annotation']['keypoints']['keypoint']:
            name = keypoint['@name']
            dictionary[name + '_v'].append(int(keypoint['@visible']))
            dictionary[name + '_x'].append(int(float(keypoint['@x'])))
            dictionary[name + '_y'].append(int(float(keypoint['@y'])))

    try:
        image = cv2.imread(os.path.join(images_dir, class_name, object_name[:-4] + '.jpeg'))
        k = show_image(image)
    except cv2.error:
        image = cv2.imread(os.path.join(images_dir, class_name, object_name[:-4] + '.jpg'))
        k = show_image(image)

    action_classes = ['stand', 'sit', 'lie', 'go', 'run', 'jump', 'interact', 'sleep', 'eat']
    for class_id in range(len(action_classes)):
        if k - 49 == class_id:
            dictionary[action_classes[class_id]].append(1)
        else:
            dictionary[action_classes[class_id]].append(0)

    cv2.imwrite('data/animals/' + dictionary['id'][-1] + '.jpeg', image)

    index += 1

df = pd.DataFrame.from_dict(dictionary)
df.to_csv("data/animals_data_" + class_name + '.csv')
