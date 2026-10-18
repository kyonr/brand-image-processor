import io
import os
import zipfile
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
    # リクエストボディ（生データ）を直接取得
    body = await request.body()
    
    if not body:
        raise HTTPException(status_code=400, detail="データが届いていません")
    
    images = []
    
    # 1. ZIPデータとして解凍を試みる
    try:
        with zipfile.ZipFile(io.BytesIO(body)) as z:
            for filename in z.namelist():
                if not filename.startswith("__MACOSX") and not filename.startswith("."):
                    img_data = z.read(filename)
                    try:
                        images.append(Image.open(io.BytesIO(img_data)))
                    except Exception:
                        pass
    except Exception:
        # 2. ZIPでない場合は単一画像として試す
        try:
            images.append(Image.open(io.BytesIO(body)))
        except Exception:
            pass

    if not images:
        raise HTTPException(status_code=400, detail="画像データを読み込めませんでした")

    # 3. 露出補正（明るさ+10%）
    processed_images = []
    for img in images:
        img = img.convert("RGB")
        enhancer = ImageEnhance.Brightness(img)
        processed_images.append(enhancer.enhance(1.10))

    # 4. グリッド結合（コラージュ作成）
    collage = create_grid_collage(processed_images)
    
    # 5. 完成画像をJPEGバイナリとして返却
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
