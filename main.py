import io
import json
import math
import os
import zipfile
from fastapi import FastAPI, HTTPException, Request, Response
from google import genai
from google.genai import types
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

app = FastAPI()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None


@app.get("/")
def read_root():
  return {
      "status": (
          "Brand Image Processor API (Independent Dual Collage & Ultra Black"
          " Protection) is running"
      )
  }


# === 精密な黒・暗色系商品の自動判定関数 ===
def is_dark_item(img: Image.Image) -> bool:
  """白背景を完全に排除し、商品本体が黒・紺・ダークカラーか判定"""
  w, h = img.size
  # 中央50%の領域（確実に商品が存在する部分）
  crop_area = img.crop((int(w * 0.25), int(h * 0.25), int(w * 0.75), int(h * 0.75)))
  gray = crop_area.convert("L")

  # 背景の白（輝度190以上）を除外した「純・商品領域」
  product_pixels = [p for p in list(gray.getdata()) if p < 190]

  if not product_pixels:
    return False

  # 商品領域の平均輝度が110未満ならダークカラーと判断
  avg_darkness = sum(product_pixels) / len(product_pixels)
  return avg_darkness < 110


# === 1. 明るさ＆色彩補正（全カラー万能 ＋ 深黒補正） ===
def auto_enhance_and_white_background(img: Image.Image) -> Image.Image:
  img = img.convert("RGB")
  dark_mode = is_dark_item(img)

  lut = []
  for i in range(256):
    x = i / 255.0

    if dark_mode:
      # 【黒・紺・ダークカラー専用】
      # 商品の黒をしっかり深みのある黒に保ち、白光り・テカリを防ぎつつ背景のみ爽やかな白にする
      if x < 0.35:
        y = math.pow(x, 1.10) * 0.90
      elif x < 0.75:
        t = (x - 0.35) / (0.75 - 0.35)
        y_35 = math.pow(0.35, 1.10) * 0.90
        y = y_35 + t * (0.85 - y_35)
      elif x < 0.92:
        t = (x - 0.75) / (0.92 - 0.75)
        y = 0.85 + math.sin(t * math.pi / 2) * (0.98 - 0.85)
      else:
        t = (x - 0.92) / 0.08
        y = 0.98 + t * 0.02
    else:
      # 【通常カラー（パステル・ブラウン・ベージュ等）高評価獲得済みの設定】
      if x < 0.68:
        y = math.pow(x, 0.52) * 1.12
      elif x < 0.92:
        t = (x - 0.68) / (0.92 - 0.68)
        y_68 = math.pow(0.68, 0.52) * 1.12
        y = y_68 + math.sin(t * math.pi / 2) * (0.98 - y_68)
      else:
        t = (x - 0.92) / 0.08
        y = 0.98 + t * 0.02

    val = min(255, max(0, int(y * 255.0)))
    lut.append(val)

  enhanced = img.point(lut * 3)

  if dark_mode:
    # 黒アイテム：コントラストを適度に効かせて重厚感を表現
    enhancer_c = ImageEnhance.Contrast(enhanced)
    enhanced = enhancer_c.enhance(1.20)
    enhancer_s = ImageEnhance.Color(enhanced)
    enhanced = enhancer_s.enhance(1.05)
  else:
    # 通常カラー
    enhancer_s = ImageEnhance.Color(enhanced)
    enhanced = enhancer_s.enhance(1.22)
    enhancer_c = ImageEnhance.Contrast(enhanced)
    enhanced = enhancer_c.enhance(1.10)

  return enhanced.filter(
      ImageFilter.UnsharpMask(radius=1.2, percent=130, threshold=2)
  )


def crop_center(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
  return ImageOps.fit(
      img,
      (target_w, target_h),
      method=Image.Resampling.LANCZOS,
      centering=(0.5, 0.5),
  )


# === 2. レイアウト生成エンジン（横一列1x4 ＆ 2x4格子） ===


def make_1x4_side_strip(images: list[Image.Image]) -> Image.Image:
  """【側面専用】1000x1000正方形キャンバスに縦長写真を必ず横一列（1x4）に並べる"""
  if not images:
    return None
  n = len(images)
  if n == 1:
    return crop_center(images[0], 1000, 1000)

  cell_w = 1000 // n
  cell_h = 1000
  collage = Image.new("RGB", (1000, 1000), (255, 255, 255))
  for idx, img in enumerate(images):
    collage.paste(crop_center(img, cell_w, cell_h), (idx * cell_w, 0))
  return collage


def make_corners_2x4_grid(
    images: list[Image.Image], target_size: int = 1000
) -> Image.Image:
  """【角スレ専用】4列x2行 (2x4) グリッドレイアウト"""
  if not images:
    return None
  n = len(images)
  if n == 1:
    return crop_center(images[0], target_size, target_size)

  if n <= 2:
    cols, rows = 2, 1
  elif n <= 4:
    cols, rows = 2, 2
  elif n <= 6:
    cols, rows = 3, 2
  else:
    cols, rows = 4, 2  # 7〜8枚：上4枚・下4枚

  cell_w = target_size // cols
  cell_h = target_size // rows

  collage = Image.new("RGB", (target_size, target_size), (255, 255, 255))
  for idx, img in enumerate(images[: cols * rows]):
    r, c = idx // cols, idx % cols
    collage.paste(crop_center(img, cell_w, cell_h), (c * cell_w, r * cell_h))
  return collage


# === 3. メインAPIエンドポイント ===
@app.post("/process-images")
async def process_images(request: Request):
  enhanced_images = []
  try:
    body = await request.body()
    if not body:
      raise HTTPException(status_code=400, detail="No data")

    images = []
    try:
      with zipfile.ZipFile(io.BytesIO(body)) as z:
        for filename in z.namelist():
          if not filename.startswith("__MACOSX") and not filename.startswith(
              "."
          ):
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
      raise HTTPException(status_code=400, detail="No images")

    # 1. 補正適用
    enhanced_images = [auto_enhance_and_white_background(img) for img in images]

    # 2. アスペクト比（縦横比）による【完全厳格分類】
    portrait_sides = []  # 縦長写真 ➔ 1x4横一列結合専用
    square_corners = []  # 四角写真 ➔ 2x4格子結合専用

    for img in enhanced_images:
      w, h = img.size
      # 縦横比が1.02を超えるものはすべて「縦長ショット（側面）」として判定
      if h > w * 1.02:
        portrait_sides.append(img)
      else:
        square_corners.append(img)

    output_files = {}

    # 3. それぞれ独立したファイルとして結合画像を生成！
    # ① 側面（縦長写真がある場合）➔ 必ず独立した 06_sides_1x4.jpg を作成
    if portrait_sides:
      output_files["06_sides_1x4.jpg"] = make_1x4_side_strip(portrait_sides)

    # ② 角・パーツ（四角写真がある場合）➔ 必ず独立した 08_corners_2x4.jpg を作成
    if square_corners:
      output_files["08_corners_2x4.jpg"] = make_corners_2x4_grid(square_corners)

    # 4. ZIP圧縮＆レスポンス生成
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
      for fname, img_obj in output_files.items():
        if img_obj:
          img_buf = io.BytesIO()
          img_obj.save(img_buf, format="JPEG", quality=95)
          zf.writestr(fname, img_buf.getvalue())

    zip_buf.seek(0)
    return Response(
        content=zip_buf.getvalue(),
        media_type="application/zip",
        headers={
            "Content-Disposition": "attachment; filename=processed_images.zip"
        },
    )

  except Exception as e:
    # エラー時でも絶対にショートカットを落とさないよう、補正済みの単品画像をZIPにして返却
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
      if enhanced_images:
        for idx, img in enumerate(enhanced_images):
          img_buf = io.BytesIO()
          img.save(img_buf, format="JPEG", quality=90)
          zf.writestr(f"{idx+1:02d}_backup_photo.jpg", img_buf.getvalue())
      else:
        zf.writestr("00_error.txt", f"Error processing: {str(e)}")

    zip_buf.seek(0)
    return Response(
        content=zip_buf.getvalue(),
        media_type="application/zip",
        headers={
            "Content-Disposition": "attachment; filename=processed_images.zip"
        },
    )
