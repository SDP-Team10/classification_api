import cv2
import numpy as np
import uvicorn

from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from classification_api.classifiers import classify_trash

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/classify/")
def make_inference(image_file: UploadFile = File(...)):
    try:
        np_image = np.fromfile(image_file.file, dtype=np.uint8)
        image = cv2.imdecode(np_image, -1)
        pred = classify_trash(image=image)
        return {"result": pred[0]}
    except:
        raise HTTPException(
            status_code=500, detail="Unable to process file"
        )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)