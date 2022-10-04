import json
import argparse
from itertools import filterfalse
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser(description='Splits COCO annotations file into training and test sets.')
parser.add_argument('annotations', metavar='coco_annotations', type=str, help='Path to COCO annotations file')
parser.add_argument('train_file', type=str, help='Path to directory where the training annotations must be stored')
parser.add_argument('valid_file', type=str, help='Path to directory where the validation annotations must be stored')
parser.add_argument('valid_frac', type=float, help='Fraction of training annotations to be used for validation')
args = parser.parse_args()

def save_coco(file, info, licenses, images, annotations, categories):
    with open(file, 'wt') as coco:
        json.dump({'info': info, 'licenses': licenses, 'images': images, 'annotations': annotations, 
            'categories': categories}, coco, indent=2, sort_keys=False)

def filter_annotations(annotations, images):
    image_ids = list(map(lambda i: int(i['id']), images))
    return list(filter(lambda a: int(a['image_id']) in image_ids, annotations))

def remove_images(annotations, images):
    images_with_annotations = list(map(lambda a: int(a['image_id']), annotations))
    return list(filterfalse(lambda i: i['id'] not in images_with_annotations, images))

def main(args):
    with open(args.annotations, 'rt', encoding='UTF-8') as annotations:
        coco_file = json.load(annotations)
        annotations = coco_file['annotations']
        categories = coco_file['categories']
        licenses = coco_file['licenses']
        images = coco_file['images']
        info = coco_file['info']

        # Estimate the number of images
        number_of_images = len(images)

        # Sort and remove the images without annotations
        images = remove_images(annotations, images)

        # Split the COCO annotations file into training and validation
        train_imgs, valid_imgs = train_test_split(images, train_size=(1.0-args.valid_frac))

        # Filter the annotations to include the training and validation annotations
        train_anns = filter_annotations(annotations, train_imgs)
        valid_anns = filter_annotations(annotations, valid_imgs)

        # Save the training and validation annotations as COCO JSON format
        save_coco(args.train_file, info, licenses, train_imgs, train_anns, categories)
        save_coco(args.valid_file, info, licenses, valid_imgs, valid_anns, categories)

        print("Saved {} annotations in {} and {} in {}".format(len(train_anns), args.train_file, len(valid_anns), args.valid_file))
        print("Saved {} images in {} and {} in {}".format(len(train_imgs), args.train_file, len(valid_imgs), args.valid_file))

if __name__ == "__main__":
    main(args)