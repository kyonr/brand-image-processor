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
    # 1. 送信されたフォームデータを取得
    form = await request.form()
    
    images = []
    # 2. 届いたすべてのキー・データから画像ファイルを自動検出して抽出
    for key in form.keys():
        items = form.getlist(key)
        for item in items:
            if hasattr(item, "read"):
                contents = await item.read()
                if contents and len(contents) > 0:
                    try:
                        img = Image.open(io.BytesIO(contents))
                        images.append(img)
                    except Exception:
                        pass

    if not images:
        raise HTTPException(
            status_code=400, 
            detail=f"画像が受信できませんでした。（受信キー: {list(form.keys())}）"
        )

    # 3. 露出補正（明るさ調整 +10%）
    processed_images = []
    for img in images:
        img = img.convert("RGB")
        enhancer = ImageEnhance.Brightness(img)
        img_bright = enhancer.enhance(1.10)
        processed_images.append(img_bright)

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
