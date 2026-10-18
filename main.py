import io
import os
from fastapi import FastAPI, Request, HTTPException, Response
from google import genai
from PIL import Image, ImageEnhance

app = FastAPI()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

@app.get("/")
def read_root():
    return {"status": "Brand Image Processor API is running"}

@app.post("/process-images")
async def process_images(request: Request):
    # 1. iOSショートカットからのフォームデータを安全に受信
    form = await request.form()
    uploaded_files = form.getlist("files")
    
    if not uploaded_files:
        raise HTTPException(status_code=400, detail="画像が送信されていません")
    
    images = []
    for file_item in uploaded_files:
        if hasattr(file_item, "read"):
            contents = await file_item.read()
            if contents:
                images.append(Image.open(io.BytesIO(contents)))
    
    if not images:
        raise HTTPException(status_code=400, detail="有効な画像データが含まれていません")

    # 2. 露出補正（明るさ調整 +10%）
    processed_images = []
    for img in images:
        img = img.convert("RGB")
        enhancer = ImageEnhance.Brightness(img)
        img_bright = enhancer.enhance(1.10)  # 明るさ+10%
        processed_images.append(img_bright)

    # 3. 画像のグリッド結合（コラージュ作成）
    collage = create_grid_collage(processed_images)
    
    # 4. 完成画像をJPEGバイナリとして返却
    buf = io.BytesIO()
    collage.save(buf, format="JPEG", quality=95)
    buf.seek(0)
    
    return Response(content=buf.getvalue(), media_type="image/jpeg")

def create_grid_collage(images):
    n = len(images)
    cols = 2 if n <= 4 else 4
    rows = (n + cols - 1) // cols
    
    cell_w, cell_h = 800, 800
    collage = Image.new("RGB", (cols * cell_w, rows * cell_h), (255, 255, 255))
    
    for idx, img in enumerate(images):
        r = idx // cols
        c = idx % cols
        img_resized = img.resize((cell_w, cell_h))
        collage.paste(img_resized, (c * cell_w, r * cell_h))
        
    return collage
