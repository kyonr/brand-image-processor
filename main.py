import io
import math
import os
import zipfile
from fastapi import FastAPI, HTTPException, Request, Response
from PIL import Image, ImageEnhance, ImageOps

app = FastAPI()


@app.get("/")
def read_root():
  return {"status": "Brand Image Processor API is running"}


def auto_enhance_and_white_background(img: Image.Image) -> Image.Image:
  """ガンマ補正で黒い商品（暗部）の質感やシボ感を明るく持ち上げ、背景だけを真っ白にする処理"""
  img = img.convert("RGB")

  # ⚙️ ガンマ値（小さくするほど黒い革の質感・柄が明るく浮き出ます）
  GAMMA = 0.68

  lut = []
  for i in range(256):
    norm = i / 255.0
    # 1. ガンマ補正で暗い領域（黒いレザーの質感など）をぐっと明るく持ち上げる
    g_val = math.pow(norm, GAMMA) * 255.0

    # 2. 背景の薄暗い影（ハイライト領域）をすっきり真っ白（255）へ補正
    if g_val > 170:
      g_val = 255.0
    elif g_val > 130:
      # 130〜170の間を滑らかに255へ繋ぐ
      ratio = (g_val - 130) / (170 - 130)
      g_val = 130 + ratio * (255 - 130)

    lut.append(min(255, max(0, int(g_val))))

  enhanced = img.point(lut * 3)

  # コントラストを微調整して金具や輪郭にメリハリをつける
  enhancer_c = ImageEnhance.Contrast(enhanced)
  enhanced = enhancer_c.enhance(1.10)

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

  # 1. ガンマ補正で黒い商品の質感浮き出し ＆ 背景白飛び
  processed_images = [auto_enhance_and_white_background(img) for img in images]

  # 2. 自動レイアウト生成
  collage = create_optimal_collage(processed_images)

  buf = io.BytesIO()
  collage.save(buf, format="JPEG", quality=95)
  buf.seek(0)

  return Response(content=buf.getvalue(), media_type="image/jpeg")
