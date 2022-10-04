import os
import json
import sleap
import argparse
import numpy as np
from PIL import Image
from datetime import datetime

# This prevents TensorFlow from allocating all the GPU memory, which leads to issues on
# some GPUs/platforms:
sleap.disable_preallocation()

# This would hide GPUs from the TensorFlow altogether:
# sleap.use_cpu_only()

# Print some info:
# sleap.versions()
# sleap.system_summary()

parser = argparse.ArgumentParser(description='Convert Sleap labels package to COCO Keypoint dataset.')
parser.add_argument("labels_package", type=str, help="Path to input .pkg.slp file.")
parser.add_argument("coco_json", type=str, help="Path to output .json file.")
parser.add_argument("image_dir", default="", help="Path to video directory")
args = parser.parse_args()

# Load the label from the Sleap dataset from a .slp file
labels = sleap.Labels.load_file(args.labels_package)

# Compute the current date
current_date = datetime.today().strftime('%Y-%m-%d')
current_year = datetime.today().strftime('%Y')

# Info and license
info = {
    'year': current_year,
    'version': "v1.0",
    'description': "",
    'contributor': "Alexander Tubæk Rasmussen",
    'date_created': current_date,
}

license = {
    'id': 0,
    'name': "MIT License",
    'url': "https://opensource.org/licenses/MIT"
}

# Compute the categories
skeletons = labels.skeletons
if skeletons == 1:
        print(f"This code is created for sleap labels packages containing one skeleton only.")

categories, category_id = [], 0
category_name = "harbor porpoise"
nodes, edges = [], []
skeleton_nodes = skeletons[0].nodes
skeleton_edges = skeletons[0].edges
for i in range(len(skeleton_nodes)):
    node = skeleton_nodes[i].name
    nodes.append(node)
for j in range(len(skeleton_edges)):
    edge = skeleton_edges[j]
    edge = [nodes.index(edge[0].name), nodes.index(edge[1].name)]
    edges.append(edge)

category = {
    'id': category_id,
    'name': category_name,
    'keypoints': nodes,
    'skeleton': edges,
}
categories.append(category)

# Compute the images and annotations along with saving the images
# https://github.com/talmolab/sleap/discussions/779
annotations, images = [], []
annotation_id, image_id = 0, 0
labeled_frames = labels.labeled_frames
for labeled_frame in labeled_frames:
    # Create and save images
    image_name = "image_" + str(image_id) + ".png"
    image_array = labeled_frame.image
    image = Image.fromarray(image_array)
    image_height = image.height
    image_width = image.width
    image.save(args.image_dir + image_name)
    image = {
        'id': image_id,
        'width': image_width,
        'height': image_height,
        'file_name': image_name,
        'date_captured': current_date,
    }
    images.append(image)
    print(f"Image id: {image_id}")

    # Create annotations
    instances = labeled_frame.instances
    for instance in instances:
        # Keypoints and number of keypoints
        keypoints, num_keypoints = [], 0
        for k in range(len(instance.points)):
            coordinate_x = instance.points[k].x
            coordinate_y = instance.points[k].y
            if instance.points[k].visible:
                visibility_flag = 2
            else:
                visibility_flag = 1
            if visibility_flag > 0:
                num_keypoints += 1
            keypoints.append(coordinate_x)
            keypoints.append(coordinate_y)
            keypoints.append(visibility_flag)
        
        # Bounding box and bounding box area
        points = instance.numpy() 
        if np.isnan(points).all():
            bbox = np.array([np.nan, np.nan, np.nan, np.nan])
        else:
            bbox = np.concatenate([np.nanmin(points, axis=0)[::-1], np.nanmax(points, axis=0)[::-1]])

        y1, x1, y2, x2 = bbox
        cx, cy = np.array([(x2 + x1) / 2, (y2 + y1) / 2])
        w, h = np.array([x2 - x1, y2 - y1])
        bbox = np.array([cx, cy, w, h])
        area = bbox[2] * bbox[3]
        annotation = {
            'id': annotation_id,
            'image_id': image_id,
            'category_id': category_id,
            'keypoints': keypoints,
            'num_keypoints': num_keypoints,
            'segmentation': [],
            'bbox': bbox,
            'area': area,
            'iscrowd': 0
        }
        annotations.append(annotation)
        annotation_id += 1
    image_id += 1

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

data = {
    'info': info,
    'licenses': [license],
    'categories': categories,
    'images': images,
    'annotations': annotations
}

# Save the coco annotations file
with open(args.coco_json, 'w') as coco_file:
    json.dump(data, coco_file, indent=2, cls=NumpyEncoder)

# python sleap_training_data/convert.py sleap_training_data/labels_packages/labels_with_images.pkg.slp images/example.json images/example/
# https://www.immersivelimit.com/create-coco-annotations-from-scratch
# https://cocodataset.org/#format-data
