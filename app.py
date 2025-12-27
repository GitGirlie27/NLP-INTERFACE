# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from fastapi.middleware.cors import CORSMiddleware
from nlp_utils import (
    lowercase, remove_punct_num, remove_stopwords,
    stemming, lemmatization, pos_tagging, bag_of_words, tf_idf
)

app = FastAPI()
@app.get("/")
def home():
    return {"message": "NLP is running — go to /docs to use it!"}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ProcessRequest(BaseModel):
    text: str
    steps: List[str] = []

class VectorizeRequest(BaseModel):
    corpus: List[str]
    method: str = "tfidf"
    max_features: Optional[int] = None

@app.post("/process")
def process(req: ProcessRequest):
    text = req.text
    tokens = None
    result = {}
    for step in req.steps:
        if step == "lowercase":
            text = lowercase(text); result["lowercase"] = text
        elif step == "remove_punct":
            text = remove_punct_num(text); result["cleaned"] = text
        elif step == "tokenize":
            tokens = tokenize(text); result["tokens"] = tokens
        elif step == "remove_stopwords":
            if tokens is None:
                tokens = tokenize(text)
            tokens = remove_stopwords(tokens); result["no_stopwords"] = tokens
        elif step == "stem":
            if tokens is None:
                tokens = tokenize(text)
            tokens = stem_tokens(tokens); result["stems"] = tokens
        elif step == "lemmatize":
            result["lemmas"] = lemmatize(text)
        elif step == "pos":
            result["pos"] = pos_tag(text)
        else:
            result[f"unknown_step_{step}"] = "ignored"
    result["final"] = tokens if tokens is not None else text
    return result

@app.post("/vectorize")
def vectorize(req: VectorizeRequest):
    if req.method == "bow":
        vec, features = bow_vectorizer(req.corpus)
    elif req.method == "tfidf":
        vec, features = tfidf_vectorizer(req.corpus)
    else:
        raise HTTPException(status_code=400, detail="Invalid method. Use 'bow' or 'tfidf'.")

    return {
        "features": features,
        "matrix": vec.tolist()
    }
