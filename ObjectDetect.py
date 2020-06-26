import cv2
import argparse
from helper import resize, sliding_window, image_pyramid
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import imagenet_utils
from imutils.object_detection import non_max_suppression

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help="path to the input image")
ap.add_argument('-s', '--size', type=str, default="(200, 150)", help="minimum ROI size in pixels")
ap.add_argument('-c', '--min_confidence', default=0.9, type=float, help="minimum probability to filter weak detections")
ap.add_argument('-v', '--visualize', default=-1, type=int,
                help="whether or not to show extra visualizations for debugging")
args = vars(ap.parse_args())

STEPS = 32
ROI_SIZE = eval(args['size'])
PYRAMID_SCALE = 1.5
WIDTH = 600
INPUT_SIZE = (224, 224)

print('[INFO] Loading Model....')
model = ResNet50(weights='imagenet', include_top=True)

org_img = cv2.imread(args['image'])
org_img = resize(org_img, width=WIDTH)
(H, W) = org_img.shape[:2]

pyramid = image_pyramid(org_img, scale=PYRAMID_SCALE, minsize=ROI_SIZE)

rois = []
locs = []

for image in pyramid:
    scale = W / float(image.shape[1])
    for (x, y, roi_org) in sliding_window(image, STEPS, ROI_SIZE):
        X = int(x * scale)
        Y = int(y * scale)
        w = int(ROI_SIZE[0] * scale)
        h = int(ROI_SIZE[1] * scale)

        roi = cv2.resize(roi_org, INPUT_SIZE)
        roi = img_to_array(roi)
        roi = preprocess_input(roi)
        rois.append(roi)
        locs.append((X, Y, X + w, Y + h))

        if args['visualize'] > 0:
            clone = org_img.copy()
            cv2.rectangle(clone, (X, Y), (X + w, Y + h), (0, 255, 255), 2)
            cv2.imshow("visualization", clone)
            cv2.imshow("ROI", roi_org)
            cv2.waitKey(0)

print('[INFO] looping over pyramids/sliding completed..')
rois = np.array(rois, dtype='float32')

print('[INFO] classifying ROIs....')
preds = model.predict(rois)
print('[INFO] classification of ROIs completed')

preds = imagenet_utils.decode_predictions(preds, top=1)

labels = {}

for (i, p) in enumerate(preds):
    (imagenet_ID, label, score) = p[0]
    if score >= args['min_confidence']:
        box = locs[i]
        L = labels.get(label, [])
        L.append((box, score))
        labels[label] = L
print('[INFO] Showing Results....')

for label in labels.keys():
    clone = org_img.copy()
    for (box, score) in labels[label]:
        (start_x, start_y, end_x, end_y) = box
        cv2.rectangle(clone, (start_x, start_y), (end_x, end_y), (0, 255, 255), 2)
        cv2.imshow("Before", clone)

    boxes = np.array([p[0] for p in labels[label]])
    probs = np.array([p[1] for p in labels[label]])
    boxes = non_max_suppression(boxes, probs)
    for (startX, startY, endX, endY) in boxes:
        cv2.rectangle(clone, (startX, startY), (endX, endY), (0, 255, 0), 2)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.putText(clone, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
    cv2.imshow("after", clone)
    cv2.waitKey(0)
cv2.destroyAllWindows()
