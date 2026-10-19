import io
import math
import os
import zipfile
from fastapi import FastAPI, HTTPException, Request, Response
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

app = FastAPI()


@app.get("/")
def read_root():
  return {"status": "Brand Image Processor API is running"}


def auto_enhance_and_white_background(img: Image.Image) -> Image.Image:
  """ガンマ補正 ＋ アンシャープマスクで革のシボ感・凹凸・輪郭をパキッと立体的に際立たせる"""
  img = img.convert("RGB")

  # 1. ガンマ補正（0.62で革の暗い質感をしっかりと浮き出させます）
  GAMMA = 0.62

  lut = []
  for i in range(256):
    norm = i / 255.0
    g_val = math.pow(norm, GAMMA) * 255.0

    # 背景の白飛び処理
    if g_val > 165:
      g_val = 255.0
    elif g_val > 125:
      ratio = (g_val - 125) / (165 - 125)
      g_val = 125 + ratio * (255 - 125)

    lut.append(min(255, max(0, int(g_val))))

  enhanced = img.point(lut * 3)

  # 2. コントラストの補正（黒を引き締めつつ陰影を作る）
  enhancer_c = ImageEnhance.Contrast(enhanced)
  enhanced = enhancer_c.enhance(1.15)

  # 3. アンシャープマスク処理（凹凸・シボ感・金属パーツの輪郭をパキッとシャープ化）
  enhanced = enhanced.filter(
      ImageFilter.UnsharpMask(radius=2.0, percent=160, threshold=3)
  )

  return enhanced


def crop_center(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
  """中央の被写体を残して綺麗にリサイズ・トリミング"""
  return ImageOps.fit(
      img,
      (target_w, target_h),
      method=Image.Resampling.LANCZOS,
      centering=(0.5, 0.5),
  )


def create_optimal_collage(images: list[Image.Image]) -> Image.Image:
  """枚数に合わせて最適なプロ仕様レイアウトを自動生成"""
  n = len(images)
  if n == 0:
    return None

  # --- 4枚の場合：1行4列（縦分割）に自動レイアウト ---
  if n == 4:
    cell_w, cell_h = 350, 1100
    collage = Image.new("RGB", (cell_w * 4, cell_h), (255, 255, 255))
    for idx, img in enumerate(images):
      fitted_img = crop_center(img, cell_w, cell_h)
      collage.paste(fitted_img, (idx * cell_w, 0))
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
    cell_w, cell_h = 450, 800
    collage = Image.new("RGB", (cell_w * 3, cell_h), (255, 255, 255))
    for idx, img in enumerate(images):
      fitted_img = crop_center(img, cell_w, cell_h)
      collage.paste(fitted_img, (idx * cell_w, 0))
    return collage

  # --- 5枚以上など（デフォルト） ---
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

  # 1. 自動明るさ・輪郭くっきりシャープ化・背景白飛び
  processed_images = [auto_enhance_and_white_background(img) for img in images]

  # 2. 自動レイアウト生成
  collage = create_optimal_collage(processed_images)

  buf = io.BytesIO()
  collage.save(buf, format="JPEG", quality=95)
  buf.seek(0)

  return Response(content=buf.getvalue(), media_type="image/jpeg")
