import os
import sys
from fastapi import FastAPI
from scripts.utility import get_score

sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/scripts")

app = FastAPI()


@app.get("/")
def root():
    return {"message": "Hello World"}


@app.get("/api/v1/score/{src1}/{src2}")
def func(src1: str, src2: str):
    score = get_score(src1, src2)
    return {"score": score}