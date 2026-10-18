import io
import os
import zipfile
from fastapi import FastAPI, Request, HTTPException, Response
from google import genai
from PIL import Image, ImageEnhance, ImageOps

app = FastAPI()

# ==========================================
# ⚙️ 画像調整パラメータ
# ==========================================
BRIGHTNESS_FACTOR = 1.25  # 明るさ（1.25 = 25%アップ）
CONTRAST_FACTOR   = 1.20  # コントラスト（1.20 = 20%アップで黒を引き締める）
WHITE_THRESHOLD   = 190   # 背景白飛ばしの閾値（この値以上の背景を真っ白に補正）

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

@app.get("/")
def read_root():
    return {"status": "Brand Image Processor API is running"}

def enhance_image_for_ec(img: Image.Image) -> Image.Image:
    """商品画像用に明るさ・コントラスト・背景白飛ばしを適用する処理"""
    img = img.convert("RGB")
    
    # 1. コントラスト補正（黒を引き締める）
    enhancer_c = ImageEnhance.Contrast(img)
    img = enhancer_c.enhance(CONTRAST_FACTOR)
    
    # 2. 明るさ補正
    enhancer_b = ImageEnhance.Brightness(img)
    img = enhancer_b.enhance(BRIGHTNESS_FACTOR)
    
    # 3. 背景白飛ばしトーンカーブ処理
    lut = []
    for i in range(256):
        if i < 40:
            val = i * 0.95
        elif i > WHITE_THRESHOLD:
            val = i + (255 - i) * 0.7
        else:
            val = i
        lut.append(min(255, max(0, int(val))))
    
    return img.point(lut * 3)

def crop_center(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    """画像を中央基準でターゲットサイズに合わせて自動トリミング・リサイズ"""
    return ImageOps.fit(img, (target_w, target_h), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))

def create_optimal_collage(images: list[Image.Image]) -> Image.Image:
    """枚数や写真のアスペクト比（縦横比）から最適なレイアウトを自動生成"""
    n = len(images)
    if n == 0:
        return None
    
    # 送信された写真の平均アスペクト比（幅 / 高さ）を計算
    aspect_ratios = [img.width / img.height for img in images]
    avg_aspect = sum(aspect_ratios) / n
    
    # --- 4枚の場合のレイアウト自動分岐 ---
    if n == 4:
        # 写真が縦向き（アスペクト比 < 0.95）の場合 ➔ 見本通りの「1行4列（縦分割）」
        if avg_aspect < 0.95:
            cell_w, cell_h = 300, 1000  # 縦長セル（4枚並ぶと綺麗な正方形〜横長）
            collage = Image.new("RGB", (cell_w * 4, cell_h), (255, 255, 255))
            for idx, img in enumerate(images):
                fitted_img = crop_center(img, cell_w, cell_h)
                collage.paste(fitted_img, (idx * cell_w, 0))
            return collage
        
        # 写真が正方形〜横長の場合 ➔ 「2行2列（格子状）」
        else:
            cell_w, cell_h = 800, 800
            collage = Image.new("RGB", (cell_w * 2, cell_h * 2), (255, 255, 255))
            for idx, img in enumerate(images):
                fitted_img = crop_center(img, cell_w, cell_h)
                r, c = idx // 2, idx % 2
                collage.paste(fitted_img, (c * cell_w, r * cell_h))
            return collage

    # --- 2枚の場合 ---
    elif n == 2:
        cell_w, cell_h = 600, 800
        collage = Image.new("RGB", (cell_w * 2, cell_h), (255, 255, 255))
        for idx, img in enumerate(images):
            fitted_img = crop_center(img, cell_w, cell_h)
            collage.paste(fitted_img, (idx * cell_w, 0))
        return collage

    # --- 3枚の場合 ---
    elif n == 3:
        cell_w, cell_h = 400, 800
        collage = Image.new("RGB", (cell_w * 3, cell_h), (255, 255, 255))
        for idx, img in enumerate(images):
            fitted_img = crop_center(img, cell_w, cell_h)
            collage.paste(fitted_img, (idx * cell_w, 0))
        return collage

    # --- その他（デフォルト） ---
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

    # 3. 商品画像向け補正処理（明るさ・コントラスト・白飛ばし）
    processed_images = [enhance_image_for_ec(img) for img in images]

    # 4. 最適レイアウトでの自動コラージュ結合
    collage = create_optimal_collage(processed_images)
    
    # 5. 完成画像をJPEGバイナリとして返却
    buf = io.BytesIO()
    collage.save(buf, format="JPEG", quality=95)
    buf.seek(0)
    
    return Response(content=buf.getvalue(), media_type="image/jpeg")
