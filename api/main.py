from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from ml.inference import load_model, predict_image
from PIL import Image

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model, label_map = load_model()

@app.get("/")
def root():
    return {"message": "Mindsight ML Demo API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(file.file).convert("RGB")
    prediction, confidence = predict_image(image, model, label_map)

    return {
        "prediction": prediction,
        "confidence": confidence
    }