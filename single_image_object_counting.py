
import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import cv2
import coco
import utils
import model as modellib


# PART - 1: A introduction to using the pre-trained model to detect and segment objects.

# Root directory of the project
ROOT_DIR = os.getcwd()


MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file

COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)


# PART - 2: (Mainly about configuration) We'll be using a model trained on the MS-COCO dataset. The configurations of this model are in the ```CocoConfig``` class in ```coco.py```. 

class InferenceConfig(coco.CocoConfig):

    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2

config = InferenceConfig()
config.display()


# PART - 3: Create Model and Load Trained Weights



# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)


# PART - 4: # COCO Class names, Index of the class in the list is its ID. For example, to get ID of the teddy bear class, 
# use: class_names.index('teddy bear')

class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']


# PART - 5: Mask R-CNN 

import random
import itertools
import colorsys
import numpy as np
from skimage.measure import find_contours
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines
from matplotlib.patches import Polygon
import IPython.display
import cv2

import utils



#  PART - 5.1:Visualization functions



def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image



def get_masked_fixed_color(image, boxes, masks, class_ids, class_names,
                      colors = None, scores=None, title="",
                      figsize=(16, 16), ax=None, show=True):

    objects = dict()
    
    # Number of instances
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    # Generate random colors
    if colors == None:
        classN = len(class_names)
        colors = random_colors(classN)

    masked_image = np.array(image)

    for i in range(N):
        color = colors[class_ids[i]]

        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        y1, x1, y2, x2 = boxes[i]
        cv2.rectangle(masked_image, (x1, y1), (x2, y2), (0,0,255), thickness = 2)

        # Label
        class_id = class_ids[i]
        score = scores[i] if scores is not None else None
        label = class_names[class_id]
        if(label in objects):
            objects[label] += 1
        else:
            objects[label] = 1
        x = random.randint(x1, (x1 + x2) // 2)
        caption = "{} {:.3f}".format(label, score) if score else label
        cv2.putText(masked_image, caption, (x1 + 5, y1 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255))

        # Mask
        mask = masks[:, :, i]
        if show: 
            masked_image = apply_mask(masked_image, mask, color)

            # Mask Polygon
            # Pad to ensure proper polygons for masks that touch image edges.
            padded_mask = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
            padded_mask[1:-1, 1:-1] = mask
            contours = find_contours(padded_mask, 0.5)
            for verts in contours:
                # Subtract the padding and flip (y, x) to (x, y)
                verts = np.fliplr(verts) - 1
                verts = verts.reshape((-1, 1, 2)).astype(np.int32)
                # Draw an edge on object contour
                cv2.polylines(masked_image, verts, True, color)

    print(str(objects))
    cv2.putText(masked_image, str(objects), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0))
    retobj=[objects,masked_image]
    return retobj


#  PART - 5.2: Object detection is powered by OpenCV


import cv2
import time
import glob
import sys
import csv
colors = random_colors(len(class_names))

#image1 = cv2.imread('input_images_and_videos/input.png')
images = [cv2.imread(file) for file in glob.glob("input_images_and_videos/*.png")]
#image1 = cv2.resize(image1, None, fx=0.5, fy=0.5)        
#image_batch = [image1]
        
# Run detection
t = time.time()
print(len(images))
results = model.detect(images, verbose=0)
t = t - time.time()
print (t)


          
masked_image_batch = []
# Visualize results
r = results[0]
t = time.time()
f=True
l=[]

print(len(results))
for i in range(len(results)):
    r = results[i]
    im = images[i]
    vehicle_res = get_masked_fixed_color(im, r['rois'], r['masks'], r['class_ids'], class_names, colors, r['scores'], show=False)
    count=0
    dic_res=vehicle_res[0]
    a=[]
    a.append(sys.argv[i+1])
    for x in ['bicycle', 'car', 'motorcycle','bus','truck'] :
      if x in dic_res.keys():
        count+=dic_res[x]
        a.append(dic_res[x])
      else:
        a.append(0)
    a.append(count)
    vehicle_count.append(count)
    masked_image=vehicle_res[1]
    masked_image = cv2.resize(masked_image, None, fx=3, fy=3)
    masked_image_batch.append(masked_image)
    l.append(a)

t = t - time.time()
print (t)
print(vehicle_count);

#Row for CSV. Take timestamp from arguments


print("result lists",l[0],l[1])

#write into CSV
#If testing, after each test run delete the data.csv file and create a new data.csv file in its place
with open('data.csv', 'a') as f_object:

  writer_obj=csv.writer(f_object)
  writer_obj.writerow(l[0])
  writer_obj.writerow(l[1])
  f_object.close()

#cv2.imwrite("result.png", masked_image_batch[0])
