import os
import sys
from fastapi import FastAPI
from scripts.utility import get_score, get_feature, normalization

sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/scripts")

app = FastAPI()


@app.get("/")
def root():
    return {"message": "Hello World"}


@app.get("/api/v1/ai/score/{src1}/{src2}")
def func(src1: str, src2: str):
    original = f"/data/images/orig/{src1}.jpg"
    target = f"/data/images/upload/orig/{src2}.jpg"
    score, filename = get_score(original, target)
    return {"score": score, "filename": f"{filename}.jpg"}

@app.get("/api/v1/ai/joint/{src}")
def get_joint(src: str):
    path = f"/data/images/orig/{src}.jpg"
    name, keypoints, bbox = get_feature(path)
    normalized_keypoint = normalization(keypoints, bbox)
    x,y,w,h = bbox
    data = {"normalized_keypoint" : normalized_keypoint, "width":w, "height":h}
    return data
