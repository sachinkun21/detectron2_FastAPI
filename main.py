from typing import Optional
from fastapi import FastAPI
from pydantic import  BaseModel
from detectron import  predict_mask

import cv2
import base64
import time
import numpy as np

app = FastAPI()


@app.get('/')
def home():
    return {"Welcome" : "to FastAPI"}


@app.get("/detectron_person/{path}")
def pred_image(b64_str: str, q: Optional[str] = None):
    try:

        start = time.time()
        b64_img = predict_mask(path)
        end = time.time()
        return {"message":path.split('.')[0]+" Masked and Saved","time_take":round(end-start,2), "b64_img": b64_img}

    except exception as e:
        return {"error" : str(e)}


class path_it(BaseModel):
    image_location: str



@app.get("/path_tobase64/")
def display_path(path_to_image: path_it):
    path = path_to_image.image_location
    img = cv2.imread(path)
    _, buffer = cv2.imencode('.jpg', img)
    b64_img = base64.b64encode(buffer).decode('utf-8')

    return {"path": path, "encoded_string":b64_img}

@app.post("/conv_b64/")
def display_path(base64_data: path_it):
    # base6 to image
    #rint(base64_data.image_location)
    base64_data = base64_data.image_location
    print(base64_data)
    nparr = np.fromstring(base64.b64decode(base64_data), np.uint8)
    img =  cv2.imdecode(nparr, cv2.IMREAD_ANYCOLOR)

    path_to_save = "b64img.jpg"
    cv2.imwrite(path_to_save, img)

    # base image to base_64
    _, buffer = cv2.imencode('.jpg', img)
    b64_img = base64.b64encode(buffer).decode('utf-8')

    return {"path": path_to_save , "encoded_string":b64_img}

# from path to b64
# def to_image_string(image_filepath):
#     return open(image_filepath, 'rb').read().encode('base64')