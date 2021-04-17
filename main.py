from typing import Optional

from fastapi import FastAPI, File, UploadFile

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/import_file")
async def import_file_post(file: UploadFile = File(...)):
    return {"filename": file.filename}