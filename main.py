from typing import  Optional
from fastapi import FastAPI

from detectron import  predict_mask

import cv2
import base64
import time

app = FastAPI()

@app.get('/')
def home():
    return {"Welcome":"to FastAPI"}

@app.get("/detectron2/{path}")
def display_path(path: str, q: Optional[str] = None):
    try:
        start = time.time()
        b64_img = predict_mask(path)
        end = time.time()
        return {"message":path.split('.')[0]+" Masked and Saved","time_take":round(end-start,2), "b64_img": b64_img}

    except exception as e:
        return {"error" : str(e)}



@app.get("/detectron/{path}")
def display_path(path: str):

    img = cv2.imread('elon.jpg')
    cv2.imwrite("elon2.jpg", img)
    _, buffer = cv2.imencode(".jpg", img)
    b64_img = base64.b64encode(buffer).decode('utf-8')

    return {"path": path, "encoded_string":b64_img}
