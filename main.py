import io
import os
import zipfile
from fastapi import FastAPI, Request, HTTPException, Response
from google import genai
from PIL import Image, ImageEnhance, ImageOps

app = FastAPI()

# ==========================================
# ⚙️ 画像調整パラメータ（安全で自然な補正）
# ==========================================
BRIGHTNESS_FACTOR = 1.15  # 明るさ（+15%）
CONTRAST_FACTOR   = 1.15  # コントラスト（+15%で黒を引き締める）

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

@app.get("/")
def read_root():
    return {"status": "Brand Image Processor API is running"}

def enhance_image_for_ec(img: Image.Image) -> Image.Image:
    """過度な白飛ばしをせず、商品が引き立つ自然な明るさに補正"""
    img = img.convert("RGB")
    
    # 1. 明るさ調整
    enhancer_b = ImageEnhance.Brightness(img)
    img = enhancer_b.enhance(BRIGHTNESS_FACTOR)
    
    # 2. コントラスト調整
    enhancer_c = ImageEnhance.Contrast(img)
    img = enhancer_c.enhance(CONTRAST_FACTOR)
    
    return img

def crop_center(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    """画像を中央基準で綺麗にリサイズ・トリミング"""
    return ImageOps.fit(img, (target_w, target_h), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))

def create_optimal_collage(images: list[Image.Image]) -> Image.Image:
    """枚数や写真のアスペクト比（縦横比）から最適なレイアウトを自動生成"""
    n = len(images)
    if n == 0:
        return None
    
    aspect_ratios = [img.width / img.height for img in images]
    avg_aspect = sum(aspect_ratios) / n
    
    # --- 4枚の場合のレイアウト分岐 ---
    if n == 4:
        # 縦長写真4枚の場合 ➔ 「1行4列（縦分割）」
        if avg_aspect < 0.95:
            cell_w, cell_h = 500, 1000  # 幅を広げて商品が削れないよう調整
            collage = Image.new("RGB", (cell_w * 4, cell_h), (255, 255, 255))
            for idx, img in enumerate(images):
                fitted_img = crop_center(img, cell_w, cell_h)
                collage.paste(fitted_img, (idx * cell_w, 0))
            return collage
        
        # 正方形〜横長写真 ➔ 「2行2列（格子状）」
        else:
            cell_w, cell_h = 800, 800
            collage = Image.new("RGB", (cell_w * 2, cell_h * 2), (255, 255, 255))
            for idx, img in enumerate(images):
                fitted_img = crop_center(img, cell_w, cell_h)
                r, c = idx // 2, idx % 2
                collage.paste(fitted_img, (c * cell_w, r * cell_h))
            return collage

    # --- その他の枚数 ---
    cols = 2 if n <= 4 else 3
    rows = (n + cols - 1) // cols
    cell_w, cell_h = 600, 600
    collage = Image.new("RGB", (cols * cell_w, rows * cell_h), (255, 255, 255))
    for idx, img in enumerate(images):
        fitted_img = crop_center(img, cell_w, cell_h)
        r, c = idx // cols, idx % cols
        collage.paste(fitted_img, (c * cell_w, r * cell_h))
    return collage

@app.post("/process-images")
async def process_images(request: Request):
    body = await request.body()
    
    if not body:
        raise HTTPException(status_code=400, detail="データが届いていません")
    
    images = []
    
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
        try:
            images.append(Image.open(io.BytesIO(body)))
        except Exception:
            pass

    if not images:
        raise HTTPException(status_code=400, detail="画像データを読み込めませんでした")

    # 補正処理 ＆ 自動コラージュ結合
    processed_images = [enhance_image_for_ec(img) for img in images]
    collage = create_optimal_collage(processed_images)
    
    buf = io.BytesIO()
    collage.save(buf, format="JPEG", quality=95)
    buf.seek(0)
    
    return Response(content=buf.getvalue(), media_type="image/jpeg")
