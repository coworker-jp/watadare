import base64
import json
import io

from PIL import Image
from flask import Flask, request
from flask_cors import CORS

from facedb import FaceDB

app = Flask(__name__)
CORS(app)

facedb = FaceDB()


@app.route("/search", methods=['POST'])
def search():
    if request.method == "POST":
        req = request.get_json()
        base64image = req['image']
        byte_image = base64.b64decode(base64image)
        image = Image.open(io.BytesIO(byte_image))
        response = facedb.get_similarities(image)
        return json.dumps(response)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=True)
