import json
import os
import cv2
import numpy as np

images_folder = 'example'


def main(images_folder):
    for annotations in os.listdir(images_folder):
        if annotations.endswith('.json'):
            with open(os.path.join(images_folder, annotations)) as f:
                json_data = json.load(f)
            image_path = os.path.join(images_folder, annotations[:len(annotations) - 5] + ('.png'))
            points = get_annotation_data(json_data)
            if os.path.exists(image_path):
                create_binary_mask(image_path, points)
                print(f'binary mask for {image_path} created')
            else:
                print("Image is missed")


def get_annotation_data(json_content):
    """
    parse annotation json file and save polygons
    :param json_content: json loader
    :return: list of points
    """
    points = []
    for l in range(len(json_content['shapes'])):
        points.append(json_content['shapes'][l]['points'])
    return points


def create_binary_mask(image_path, points) -> np:
    """
    :param image_path: absolute path to image
    :param points: list of point
    :return: image object
    """
    img = cv2.imread(image_path, 0)
    height, width, = img.shape

    img = np.zeros((height, width, 3), dtype=np.uint8)

    for polygon in points:
        cv2.fillPoly(img, [np.array(polygon, dtype=np.int32)], color=(255, 255, 255))
    cv2.imwrite(image_path[:len(image_path) - 4] + '_mask.png', img)

    return img


if __name__ == "__main__":
    main(images_folder=images_folder)