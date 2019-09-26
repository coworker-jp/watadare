import json

import numpy as np
import redis
from facenet_pytorch import MTCNN, InceptionResnetV1
from loguru import logger
from sklearn.metrics.pairwise import cosine_distances


class FaceModel:
    def __init__(self):
        self.mtcnn = MTCNN(keep_all=False)
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval()

    def crop_image(self, image):
        return self.mtcnn(image)

    def get_embedding(self, image):
        cropped_image = self.crop_image(image)
        tensor = self.resnet(cropped_image.unsqueeze(0))
        vector = tensor.detach().numpy()
        assert vector.shape == (1, 512)
        embedding = [float(val) for val in tensor.detach().numpy().flatten()]
        return embedding


class FaceDB:
    dimension = 512
    FACE_DATA = 0

    def __init__(self, index_path='index', redis_host='localhost', redis_port=6379):
        self.face_model = FaceModel()
        self.face_db = redis.Redis(host=redis_host, port=redis_port, db=self.FACE_DATA)

        self.matrix = []
        self.matrix_id = []
        self.update_matrix()

    def insert(self, image, face_data):
        vector = self.face_model.get_embedding(image)
        face_id = face_data['id']
        face_data['vector'] = vector
        json_data = json.dumps(face_data)
        self.face_db.set(face_id, json_data)
        logger.info("Insert face_id: {}, name: {}".format(face_id, face_data['name']))

    def get(self, face_id):
        raw_face_data = self.face_db.get(face_id)
        face_data = json.loads(raw_face_data)
        return face_data

    def update_matrix(self):
        self.matrix = []
        self.matrix_id = []

        for face_id in self.face_db.keys("*"):
            face_data = self.get(face_id)
            vector = face_data['vector']
            self.matrix_id.append(face_id.decode('utf-8'))
            self.matrix.append(vector)

        self.matrix = np.asarray(self.matrix)

    def get_similarities(self, image, size=10):
        query_vector = self.face_model.get_embedding(image)
        distance = cosine_distances(self.matrix, [query_vector]).flatten()

        map_distance = {}
        for idx, face_id in enumerate(self.matrix_id):
            map_distance[face_id] = distance[idx]

        sorted_ids = sorted(map_distance.items(), key=lambda x: x[1])
        limited_ids = sorted_ids[:size]

        result = []
        for face_id, distance in limited_ids:
            face_data = self.get(face_id)
            face_data['distance'] = distance
            result.append(face_data)
        return result
