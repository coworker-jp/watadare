import base64
import pprint

import requests

with open('image/1.jpg', 'rb') as f:
    image = f.read()

b64image = base64.b64encode(image).decode("utf-8")
response = requests.post('http://0.0.0.0:8080/search', json={'image': b64image})
pprint.pprint(response.json())
