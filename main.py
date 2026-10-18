import io
import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from google import genai
from PIL import Image

app = FastAPI()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

@app.get("/")
def read_root():
    return {"status": "Brand Image Processor API is running"}

@app.post("/process-images")
async def process_images(files: list[UploadFile] = File(...)):
    if not client:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY is not configured")
    
    images = []
    for file in files:
        contents = await file.read()
        images.append(Image.open(io.BytesIO(contents)))

    return {"message": f"Successfully received {len(images)} images for processing"}
