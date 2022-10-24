import os
import sys
from fastapi import FastAPI
from scripts.utility import get_score

sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/scripts")

app = FastAPI()


@app.get("/")
def root():
    return {"message": "Hello World"}


@app.get("/api/v1/ai/score/{src1}/{src2}")
def func(src1: str, src2: str):
    original = f"/data/images/orig/{src1}.jpg"
    target = f"/data/images/upload/orig/{src2}.jpg"
    print(">>>>",original, target)
    score = get_score(original, target)
    return {"score": score}