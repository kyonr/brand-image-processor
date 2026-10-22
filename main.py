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


# === 仕切り線・外枠の太さ設定（ピクセル単位） ===
MARGIN = 6  # 画像全体の周囲の白枠
GAP = 6  # 写真同士の間の白仕切り線


# === 商品カラーの3段階高精度自動判定関数 ===
def get_item_color_category(img: Image.Image) -> str:
  """背景（白や背景物）を除外し、商品領域の明るさから 'black', 'brown', 'light' に分類"""
  w, h = img.size
  crop_area = img.crop((int(w * 0.25), int(h * 0.25), int(w * 0.75), int(h * 0.75)))
  gray = crop_area.convert("L")
  pixels = list(gray.getdata())

  product_pixels = [p for p in pixels if p < 220]

  if not product_pixels:
    return "light"

  avg_brightness = sum(product_pixels) / len(product_pixels)

  if avg_brightness < 75:
    return "black"  # 黒革・ダークカラー
  elif avg_brightness < 135:
    return "brown"  # モノグラム・ブラウン・中間色
  else:
    return "light"  # パステル・ホワイト・明るいカラー


# === なめらかなガンマ補正LUT生成関数（段差・色化けを100%防止） ===
def make_smooth_lut(gamma: float) -> list[int]:
  lut = []
  for i in range(256):
    x = i / 255.0
    y = math.pow(x, gamma)  # 完全連続な曲線
    val = min(255, max(0, int(y * 255.0)))
    lut.append(val)
  return lut * 3


# === 1. 1〜2枚目専用：表面・裏面 ===
def enhance_surface(img: Image.Image) -> Image.Image:
  img.thumbnail((1200, 1200), Image.Resampling.LANCZOS)
  img = img.convert("RGB")
  color_cat = get_item_color_category(img)

  if color_cat == "black":
    enhanced = img.point(make_smooth_lut(gamma=0.95))
    enhanced = ImageEnhance.Contrast(enhanced).enhance(1.20)
    enhanced = ImageEnhance.Brightness(enhanced).enhance(1.02)
  elif color_cat == "brown":
    enhanced = img.point(make_smooth_lut(gamma=0.75))
    enhanced = ImageEnhance.Brightness(enhanced).enhance(1.05)
    enhanced = ImageEnhance.Contrast(enhanced).enhance(1.08)
  else:
    # パステル・明色：3枚目の理想的な明るさに合わせて露出を約5%ダウン調整
    enhanced = img.point(make_smooth_lut(gamma=0.72))
    enhanced = ImageEnhance.Brightness(enhanced).enhance(1.03)
    enhanced = ImageEnhance.Contrast(enhanced).enhance(1.03)

  return enhanced.filter(
      ImageFilter.UnsharpMask(radius=1.2, percent=120, threshold=2)
  )


# === 2. 3枚目専用：側面（★基準となる理想の明るさ★） ===
def enhance_side(img: Image.Image) -> Image.Image:
  img.thumbnail((1200, 1200), Image.Resampling.LANCZOS)
  img = img.convert("RGB")
  color_cat = get_item_color_category(img)

  if color_cat == "black":
    enhanced = img.point(make_smooth_lut(gamma=0.82))
    enhanced = ImageEnhance.Brightness(enhanced).enhance(1.14)
    enhanced = ImageEnhance.Contrast(enhanced).enhance(1.15)
  else:
    enhanced = img.point(make_smooth_lut(gamma=0.68))
    enhanced = ImageEnhance.Brightness(enhanced).enhance(1.05)

  return enhanced.filter(
      ImageFilter.UnsharpMask(radius=1.2, percent=120, threshold=2)
  )


# === 3. 4枚目専用：角スレ8箇所 ===
def enhance_corner(img: Image.Image) -> Image.Image:
  img.thumbnail((1200, 1200), Image.Resampling.LANCZOS)
  img = img.convert("RGB")
  color_cat = get_item_color_category(img)

  if color_cat == "black":
    enhanced = img.point(make_smooth_lut(gamma=0.92))
    enhanced = ImageEnhance.Contrast(enhanced).enhance(1.18)
  else:
    # パステル・ブラウン：3枚目と同等の自然で揃った明るさ（露出-5%調整）
    enhanced = img.point(make_smooth_lut(gamma=0.72))
    enhanced = ImageEnhance.Brightness(enhanced).enhance(1.03)

  return enhanced.filter(
      ImageFilter.UnsharpMask(radius=1.2, percent=120, threshold=2)
  )


# === 4. 標準補正（5〜9枚目：内側・ポケット用） ===
def enhance_standard(img: Image.Image) -> Image.Image:
  img.thumbnail((1200, 1200), Image.Resampling.LANCZOS)
  img = img.convert("RGB")
  color_cat = get_item_color_category(img)

  if color_cat == "black":
    enhanced = img.point(make_smooth_lut(gamma=0.80))
  else:
    enhanced = img.point(make_smooth_lut(gamma=0.68))

  enhanced = ImageEnhance.Brightness(enhanced).enhance(1.05)
  enhanced = ImageEnhance.Contrast(enhanced).enhance(1.05)

  return enhanced.filter(
      ImageFilter.UnsharpMask(radius=1.2, percent=120, threshold=2)
  )


# === 5. 10枚目専用：金具・刻印・シリアル用（高評価のため完全キープ） ===
def enhance_hardware(img: Image.Image) -> Image.Image:
  img.thumbnail((1200, 1200), Image.Resampling.LANCZOS)
  img = img.convert("RGB")

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


# === レイアウト作成パーツ群（白仕切り線＆外枠付き） ===
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


# === メインAPI ===
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
    with zipfile.ZipFile(io.BytesIO(body)) as z:
      for filename in z.namelist():
        if not filename.startswith("__MACOSX") and not filename.startswith("."):
          images.append(Image.open(io.BytesIO(z.read(filename))))

    if not images:
      raise HTTPException(status_code=400, detail="No images")

    enhanced = []
    enhanced_hw = []
    for idx, img in enumerate(images):
      enhanced_hw.append(enhance_hardware(img))

      if idx in [0, 1]:  # 1・2枚目（表面・裏面）
        enhanced.append(enhance_surface(img))
      elif 2 <= idx <= 5:  # 3枚目（側面4方向）
        enhanced.append(enhance_side(img))
      elif 6 <= idx <= 13:  # 4枚目（角スレ8箇所）
        enhanced.append(enhance_corner(img))
      else:  # 5枚目以降（標準補正）
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
