import io
import math
import os
import zipfile
import gc
from fastapi import FastAPI, HTTPException, Request, Response
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

app = FastAPI()


@app.get("/")
def read_root():
  return {"status": "Brand Image Processor API is running"}


# === 仕切り線・外枠の太さ設定（ピクセル単位） ===
MARGIN = 6  # 画像全体の周囲の白枠
GAP = 6  # 写真同士の間の白仕切り線


# === 商品カラーの3段階高精度自動判定関数 ===
def get_item_color_category(img: Image.Image) -> str:
  w, h = img.size
  crop_area = img.crop((int(w * 0.25), int(h * 0.25), int(w * 0.75), int(h * 0.75)))
  gray = crop_area.convert("L")
  pixels = list(gray.getdata())

  product_pixels = [p for p in pixels if p < 220]

  if not product_pixels:
    return "light"

  avg_brightness = sum(product_pixels) / len(product_pixels)

  if avg_brightness < 75:
    return "black"
  elif avg_brightness < 135:
    return "brown"
  else:
    return "light"


# === 背景白抜け強化トーンカーブLUT生成関数 ===
def make_white_boost_lut(
    gamma: float, boost_strength: float = 0.25
) -> list[int]:
  lut = []
  for i in range(256):
    x = i / 255.0
    y = math.pow(x, gamma)
    if x > 0.45:
      h_factor = math.pow((x - 0.45) / 0.55, 1.3)
      y = y + (1.0 - y) * h_factor * boost_strength
    val = min(255, max(0, int(y * 255.0)))
    lut.append(val)
  return lut * 3


# === 1. 1〜2枚目専用：表面・裏面 ===
def enhance_surface(img: Image.Image) -> Image.Image:
  color_cat = get_item_color_category(img)

  if color_cat == "black":
    enhanced = img.point(make_white_boost_lut(gamma=0.92, boost_strength=0.15))
    enhanced = ImageEnhance.Contrast(enhanced).enhance(1.18)
    enhanced = ImageEnhance.Brightness(enhanced).enhance(1.05)
  elif color_cat == "brown":
    enhanced = img.point(make_white_boost_lut(gamma=0.72, boost_strength=0.20))
    enhanced = ImageEnhance.Brightness(enhanced).enhance(1.08)
    enhanced = ImageEnhance.Contrast(enhanced).enhance(1.06)
  else:
    enhanced = img.point(make_white_boost_lut(gamma=0.58, boost_strength=0.28))
    enhanced = ImageEnhance.Brightness(enhanced).enhance(1.12)
    enhanced = ImageEnhance.Contrast(enhanced).enhance(1.04)

  return enhanced.filter(
      ImageFilter.UnsharpMask(radius=1.2, percent=120, threshold=2)
  )


# === 2. 3枚目専用：側面 ===
def enhance_side(img: Image.Image) -> Image.Image:
  color_cat = get_item_color_category(img)

  if color_cat == "black":
    enhanced = img.point(make_white_boost_lut(gamma=0.82, boost_strength=0.15))
    enhanced = ImageEnhance.Brightness(enhanced).enhance(1.14)
    enhanced = ImageEnhance.Contrast(enhanced).enhance(1.15)
  else:
    enhanced = img.point(make_white_boost_lut(gamma=0.62, boost_strength=0.22))
    enhanced = ImageEnhance.Brightness(enhanced).enhance(1.06)

  return enhanced.filter(
      ImageFilter.UnsharpMask(radius=1.2, percent=120, threshold=2)
  )


# === 3. 4枚目専用：角スレ8箇所 ===
def enhance_corner(img: Image.Image) -> Image.Image:
  color_cat = get_item_color_category(img)

  if color_cat == "black":
    enhanced = img.point(make_white_boost_lut(gamma=0.92, boost_strength=0.15))
    enhanced = ImageEnhance.Contrast(enhanced).enhance(1.18)
  else:
    enhanced = img.point(make_white_boost_lut(gamma=0.62, boost_strength=0.22))
    enhanced = ImageEnhance.Brightness(enhanced).enhance(1.06)

  return enhanced.filter(
      ImageFilter.UnsharpMask(radius=1.2, percent=120, threshold=2)
  )


# === 4. 標準補正（5〜9枚目：内側・ポケット用） ===
def enhance_standard(img: Image.Image) -> Image.Image:
  color_cat = get_item_color_category(img)

  if color_cat == "black":
    enhanced = img.point(make_white_boost_lut(gamma=0.80, boost_strength=0.15))
    enhanced = ImageEnhance.Brightness(enhanced).enhance(1.05)
    enhanced = ImageEnhance.Contrast(enhanced).enhance(1.05)
  else:
    enhanced = img.point(make_white_boost_lut(gamma=0.60, boost_strength=0.22))
    enhanced = ImageEnhance.Brightness(enhanced).enhance(1.06)
    enhanced = ImageEnhance.Contrast(enhanced).enhance(1.04)

  return enhanced.filter(
      ImageFilter.UnsharpMask(radius=1.2, percent=120, threshold=2)
  )


# === 5. 10枚目専用：金具・刻印・シリアル用 ===
def enhance_hardware(img: Image.Image) -> Image.Image:
  enhanced = ImageEnhance.Brightness(img).enhance(1.15)
  enhanced = ImageEnhance.Contrast(enhanced).enhance(1.05)

  return enhanced.filter(
      ImageFilter.UnsharpMask(radius=1.0, percent=110, threshold=1)
  )


def crop_center(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
  return ImageOps.fit(
      img,
      (target_w, target_h),
      method=Image.Resampling.LANCZOS,
      centering=(0.5, 0.5),
  )


# === レイアウト作成パーツ群 ===
def make_1x4_side_strip(images: list[Image.Image]) -> Image.Image:
  if not images:
    return None
  n = len(images)
  cell_w = (1000 - MARGIN * 2 - GAP * (n - 1)) // n
  cell_h = 1000 - MARGIN * 2
  collage = Image.new("RGB", (1000, 1000), (255, 255, 255))
  for idx, img in enumerate(images):
    x = MARGIN + idx * (cell_w + GAP)
    y = MARGIN
    collage.paste(crop_center(img, cell_w, cell_h), (x, y))
  return collage


def make_4x2_corners_grid(images: list[Image.Image]) -> Image.Image:
  if not images:
    return None
  cols, rows = 2, 4
  cell_w = (1000 - MARGIN * 2 - GAP * (cols - 1)) // cols
  cell_h = (1000 - MARGIN * 2 - GAP * (rows - 1)) // rows
  collage = Image.new("RGB", (1000, 1000), (255, 255, 255))
  for idx, img in enumerate(images[:8]):
    r, c = idx // cols, idx % cols
    x = MARGIN + c * (cell_w + GAP)
    y = MARGIN + r * (cell_h + GAP)
    collage.paste(crop_center(img, cell_w, cell_h), (x, y))
  return collage


def make_2x2_grid(images: list[Image.Image]) -> Image.Image:
  if not images:
    return None
  cols, rows = 2, 2
  cell_w = (1000 - MARGIN * 2 - GAP * (cols - 1)) // cols
  cell_h = (1000 - MARGIN * 2 - GAP * (rows - 1)) // rows
  collage = Image.new("RGB", (1000, 1000), (255, 255, 255))
  for idx, img in enumerate(images[:4]):
    r, c = idx // cols, idx % cols
    x = MARGIN + c * (cell_w + GAP)
    y = MARGIN + r * (cell_h + GAP)
    collage.paste(crop_center(img, cell_w, cell_h), (x, y))
  return collage


def make_stacked_vertical(images: list[Image.Image]) -> Image.Image:
  if not images:
    return None
  n = len(images)
  cell_w = 1000 - MARGIN * 2
  cell_h = (1000 - MARGIN * 2 - GAP * (n - 1)) // n
  collage = Image.new("RGB", (1000, 1000), (255, 255, 255))
  for idx, img in enumerate(images):
    x = MARGIN
    y = MARGIN + idx * (cell_h + GAP)
    collage.paste(crop_center(img, cell_w, cell_h), (x, y))
  return collage


def make_smart_grid(
    images: list[Image.Image], target_size: int = 1000
) -> Image.Image:
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
  elif n <= 9:
    cols, rows = 3, 3
  else:
    cols, rows = 4, (n + 3) // 4

  cell_w = (target_size - MARGIN * 2 - GAP * (cols - 1)) // cols
  cell_h = (target_size - MARGIN * 2 - GAP * (rows - 1)) // rows

  collage = Image.new("RGB", (target_size, target_size), (255, 255, 255))
  for idx, img in enumerate(images):
    r, c = idx // cols, idx % cols
    x = MARGIN + c * (cell_w + GAP)
    y = MARGIN + r * (cell_h + GAP)
    collage.paste(crop_center(img, cell_w, cell_h), (x, y))
  return collage


# === メインAPI（★即時軽量化・メモリ徹底解放処理★） ===
@app.post("/process-images")
async def process_images(request: Request):
  enhanced = []
  enhanced_hw = []
  try:
    p_type = request.query_params.get("type", "wallet_small").strip()
    body = await request.body()
    if not body:
      raise HTTPException(status_code=400, detail="No data")

    images = []
    # 【メモリ対策】Zip解凍時に即座に1200pxに縮小してメモリ消費を90%カット
    with zipfile.ZipFile(io.BytesIO(body)) as z:
      for filename in z.namelist():
        if not filename.startswith("__MACOSX") and not filename.startswith("."):
          with Image.open(io.BytesIO(z.read(filename))) as temp_img:
            temp_img.thumbnail((1200, 1200), Image.Resampling.LANCZOS)
            images.append(temp_img.convert("RGB"))

    del body
    gc.collect()  # メモリ強制解放

    if not images:
      raise HTTPException(status_code=400, detail="No images")

    enhanced = []
    enhanced_hw = []
    for idx, img in enumerate(images):
      enhanced_hw.append(enhance_hardware(img))

      if idx in [0, 1]:
        enhanced.append(enhance_surface(img))
      elif 2 <= idx <= 5:
        enhanced.append(enhance_side(img))
      elif 6 <= idx <= 13:
        enhanced.append(enhance_corner(img))
      else:
        enhanced.append(enhance_standard(img))

    out = {}

    def get_slice(start, end=None, hw=False):
      source = enhanced_hw if hw else enhanced
      if end is None:
        return source[start:]
      return source[start:end]

    # --- 1. 財布（ポケット少） ---
    if p_type == "wallet_small":
      if len(enhanced) >= 1:
        out["01_surface.jpg"] = crop_center(enhanced[0], 1000, 1000)
      if len(enhanced) >= 2:
        out["02_back.jpg"] = crop_center(enhanced[1], 1000, 1000)
      if len(enhanced) >= 6:
        out["03_sides_1x4.jpg"] = make_1x4_side_strip(get_slice(2, 6))
      if len(enhanced) >= 14:
        out["04_corners_4x2.jpg"] = make_4x2_corners_grid(get_slice(6, 14))
      if len(enhanced) >= 15:
        out["05_inside.jpg"] = crop_center(enhanced[14], 1000, 1000)
      if len(enhanced) >= 16:
        out["06_pocket1.jpg"] = crop_center(enhanced[15], 1000, 1000)
      if len(enhanced) >= 17:
        out["07_pocket2.jpg"] = crop_center(enhanced[16], 1000, 1000)
      if len(enhanced) >= 18:
        out["08_pocket3.jpg"] = crop_center(enhanced[17], 1000, 1000)
      if len(enhanced) >= 19:
        out["09_accessories.jpg"] = crop_center(enhanced[18], 1000, 1000)
      if len(enhanced) >= 20:
        out["10_hardware_etc.jpg"] = make_smart_grid(get_slice(19, hw=True))

    # --- 2. 財布（ポケット中） ---
    elif p_type == "wallet_medium":
      if len(enhanced) >= 1:
        out["01_surface.jpg"] = crop_center(enhanced[0], 1000, 1000)
      if len(enhanced) >= 2:
        out["02_back.jpg"] = crop_center(enhanced[1], 1000, 1000)
      if len(enhanced) >= 6:
        out["03_sides_1x4.jpg"] = make_1x4_side_strip(get_slice(2, 6))
      if len(enhanced) >= 14:
        out["04_corners_4x2.jpg"] = make_4x2_corners_grid(get_slice(6, 14))
      if len(enhanced) >= 15:
        out["05_inside.jpg"] = crop_center(enhanced[14], 1000, 1000)
      if len(enhanced) >= 17:
        out["06_pockets_2x1.jpg"] = make_stacked_vertical(get_slice(15, 17))
      if len(enhanced) >= 18:
        out["07_pocket1.jpg"] = crop_center(enhanced[17], 1000, 1000)
      if len(enhanced) >= 19:
        out["08_pocket2.jpg"] = crop_center(enhanced[18], 1000, 1000)
      if len(enhanced) >= 20:
        out["09_accessories.jpg"] = crop_center(enhanced[19], 1000, 1000)
      if len(enhanced) >= 21:
        out["10_hardware_etc.jpg"] = make_smart_grid(get_slice(20, hw=True))

    # --- 3. 財布（ポケット多） ---
    elif p_type == "wallet_large":
      if len(enhanced) >= 1:
        out["01_surface.jpg"] = crop_center(enhanced[0], 1000, 1000)
      if len(enhanced) >= 2:
        out["02_back.jpg"] = crop_center(enhanced[1], 1000, 1000)
      if len(enhanced) >= 6:
        out["03_sides_1x4.jpg"] = make_1x4_side_strip(get_slice(2, 6))
      if len(enhanced) >= 14:
        out["04_corners_4x2.jpg"] = make_4x2_corners_grid(get_slice(6, 14))
      if len(enhanced) >= 15:
        out["05_inside.jpg"] = crop_center(enhanced[14], 1000, 1000)
      if len(enhanced) >= 19:
        out["06_pockets_2x2.jpg"] = make_2x2_grid(get_slice(15, 19))
      if len(enhanced) >= 20:
        out["07_pocket1.jpg"] = crop_center(enhanced[19], 1000, 1000)
      if len(enhanced) >= 21:
        out["08_pocket2.jpg"] = crop_center(enhanced[20], 1000, 1000)
      if len(enhanced) >= 22:
        out["09_accessories.jpg"] = crop_center(enhanced[21], 1000, 1000)
      if len(enhanced) >= 23:
        out["10_hardware_etc.jpg"] = make_smart_grid(get_slice(22, hw=True))

    # --- 4. バッグ（ポケット少） ---
    elif p_type == "bag_small":
      if len(enhanced) >= 1:
        out["01_thumb.jpg"] = crop_center(enhanced[0], 1000, 1000)
      if len(enhanced) >= 2:
        out["02_front.jpg"] = crop_center(enhanced[1], 1000, 1000)
      if len(enhanced) >= 3:
        out["03_back.jpg"] = crop_center(enhanced[2], 1000, 1000)
      if len(enhanced) >= 5:
        out["04_sides_1x2.jpg"] = make_1x4_side_strip(get_slice(3, 5))
      if len(enhanced) >= 7:
        out["05_top_bottom_2x1.jpg"] = make_stacked_vertical(get_slice(5, 7))
      if len(enhanced) >= 11:
        out["06_corners_2x2.jpg"] = make_2x2_grid(get_slice(7, 11))
      if len(enhanced) >= 13:
        out["07_handles_2x1.jpg"] = make_stacked_vertical(get_slice(11, 13))
      if len(enhanced) >= 14:
        out["08_inside.jpg"] = crop_center(enhanced[13], 1000, 1000)
      if len(enhanced) >= 16:
        out["09_pockets_2x1.jpg"] = make_stacked_vertical(get_slice(14, 16))
      if len(enhanced) >= 17:
        out["10_hardware_etc.jpg"] = make_smart_grid(get_slice(16, hw=True))

    # --- 5. バッグ（ポケット多） ---
    elif p_type == "bag_large":
      if len(enhanced) >= 1:
        out["01_thumb.jpg"] = crop_center(enhanced[0], 1000, 1000)
      if len(enhanced) >= 2:
        out["02_front.jpg"] = crop_center(enhanced[1], 1000, 1000)
      if len(enhanced) >= 3:
        out["03_back.jpg"] = crop_center(enhanced[2], 1000, 1000)
      if len(enhanced) >= 7:
        out["04_sides_top_bottom_2x2.jpg"] = make_2x2_grid(get_slice(3, 7))
      if len(enhanced) >= 11:
        out["05_corners_2x2.jpg"] = make_2x2_grid(get_slice(7, 11))
      if len(enhanced) >= 13:
        out["06_handles_2x1.jpg"] = make_stacked_vertical(get_slice(11, 13))
      if len(enhanced) >= 14:
        out["07_inside.jpg"] = crop_center(enhanced[13], 1000, 1000)
      if len(enhanced) >= 16:
        out["08_pockets_2x1.jpg"] = make_stacked_vertical(get_slice(14, 16))
      if len(enhanced) >= 20:
        out["09_pockets_2x2.jpg"] = make_2x2_grid(get_slice(16, 20))
      if len(enhanced) >= 21:
        out["10_hardware_etc.jpg"] = make_smart_grid(get_slice(20, hw=True))

    # --- 6. フリー ---
    else:
      for idx, img in enumerate(enhanced):
        out[f"{idx+1:02d}_photo.jpg"] = crop_center(img, 1000, 1000)

    if not out:
      for idx, img in enumerate(enhanced):
        out[f"{idx+1:02d}_photo.jpg"] = crop_center(img, 1000, 1000)

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
      for fname, img_obj in out.items():
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
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
      if enhanced:
        for idx, img in enumerate(enhanced):
          img_buf = io.BytesIO()
          img.save(img_buf, format="JPEG", quality=90)
          zf.writestr(f"{idx+1:02d}_backup.jpg", img_buf.getvalue())
      else:
        zf.writestr("error.txt", f"Processing error: {str(e)}")

    zip_buf.seek(0)
    return Response(
        content=zip_buf.getvalue(),
        media_type="application/zip",
        headers={
            "Content-Disposition": "attachment; filename=processed_images.zip"
        },
    )
