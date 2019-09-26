import argparse
import csv
import os

import facedb
import requests
from PIL import Image
from loguru import logger


def load_face_data(input):
    face_data_list = []
    with open(input, 'r') as f:
        reader = csv.DictReader(f)
        for field in reader:
            face_data_list.append(dict(field))
    return face_data_list


def download_image(image_path, face_id, image_url):
    logger.info("Get {}".format(image_url))
    response = requests.get(image_url)
    image_name = "{}.jpg".format(face_id)
    path = os.path.join(image_path, image_name)
    with open(path, 'wb') as f:
        logger.info("Save {}".format(path))
        f.write(response.content)
    return path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='data/face.csv')
    parser.add_argument('--image-path', default='image')
    parser.add_argument('--index-path', default='index')
    parser.add_argument('--redis-host', default='localhost')
    parser.add_argument('--redis-port', type=int, default=6379)
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.image_path):
        logger.info("mkdir {}".format(args.image_path))
        os.mkdir(args.image_path)

    face_data_list = load_face_data(args.input)
    face_db = facedb.FaceDB(
        index_path=args.index_path,
        redis_host=args.redis_host,
        redis_port=args.redis_port
    )

    for face_data in face_data_list:
        path = download_image(args.image_path, face_data['id'], face_data['image_url'])
        image = Image.open(path)
        face_db.insert(image, face_data)


if __name__ == "__main__":
    main()
