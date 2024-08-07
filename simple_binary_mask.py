import json
import cv2
import numpy as np

image_path = r'example\00011-502.png'
annotation_json_path = r'example\00011-502.json'


def main(image, annotations):

    with open(annotations) as f:
        json_data = json.load(f)
    points = get_annotation_data(json_data)
    create_binary_mask(image, points)
    print(f'binary mask for {image} created')


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
        cv2.fillPoly(img, [np.array(polygon,dtype=np.int32)], color=(255, 255, 255))
    cv2.imwrite(image_path[:len(image_path)-4] + '_mask.png', img)

    return img


if __name__ == "__main__":
    main(image=image_path, annotations=annotation_json_path)

