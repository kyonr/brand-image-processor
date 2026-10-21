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


# === 黒・暗色系商品の自動判定関数 ===
def is_dark_item(img: Image.Image) -> bool:
  w, h = img.size
  crop_area = img.crop((int(w * 0.25), int(h * 0.25), int(w * 0.75), int(h * 0.75)))
  gray = crop_area.convert("L")
  product_pixels = [p for p in list(gray.getdata()) if p < 195]
  if not product_pixels:
    return False
  avg_darkness = sum(product_pixels) / len(product_pixels)
  return avg_darkness < 95


# === 1. 標準補正（5枚目以降：大満足の設定を完全維持） ===
def enhance_standard(img: Image.Image) -> Image.Image:
  img.thumbnail((1200, 1200), Image.Resampling.LANCZOS)
  img = img.convert("RGB")
  dark_mode = is_dark_item(img)

  lut = []
  for i in range(256):
    x = i / 255.0
    if dark_mode:
      if x < 0.45:
        y = math.pow(x, 0.75) * 1.15
      elif x < 0.85:
        t = (x - 0.45) / (0.85 - 0.45)
        y_45 = math.pow(0.45, 0.75) * 1.15
        y = y_45 + t * (0.96 - y_45)
      else:
        t = (x - 0.85) / 0.15
        y = 0.96 + t * 0.04
    else:
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
    enhancer_b = ImageEnhance.Brightness(enhanced)
    enhanced = enhancer_b.enhance(1.08)
    enhancer_c = ImageEnhance.Contrast(enhanced)
    enhanced = enhancer_c.enhance(1.10)
    enhancer_s = ImageEnhance.Color(enhanced)
    enhanced = enhancer_s.enhance(1.05)
  else:
    enhancer_s = ImageEnhance.Color(enhanced)
    enhanced = enhancer_s.enhance(1.22)
    enhancer_c = ImageEnhance.Contrast(enhanced)
    enhanced = enhancer_c.enhance(1.10)

  return enhanced.filter(
      ImageFilter.UnsharpMask(radius=1.2, percent=120, threshold=2)
  )


# === 2. 1〜2枚目専用：表面・裏面（黒引き締め＆白光りカット） ===
def enhance_surface(img: Image.Image) -> Image.Image:
  img.thumbnail((1200, 1200), Image.Resampling.LANCZOS)
  img = img.convert("RGB")
  dark_mode = is_dark_item(img)

  lut = []
  for i in range(256):
    x = i / 255.0
    if dark_mode:
      if x < 0.40:
        y = math.pow(x, 0.95) * 0.98
      elif x < 0.80:
        t = (x - 0.40) / (0.80 - 0.40)
        y_40 = math.pow(0.40, 0.95) * 0.98
        y = y_40 + t * (0.95 - y_40)
      else:
        t = (x - 0.80) / 0.20
        y = 0.95 + t * 0.05
    else:
      y = x
    val = min(255, max(0, int(y * 255.0)))
    lut.append(val)

  enhanced = img.point(lut * 3)

  if dark_mode:
    enhancer_c = ImageEnhance.Contrast(enhanced)
    enhanced = enhancer_c.enhance(1.20)
    enhancer_b = ImageEnhance.Brightness(enhanced)
    enhanced = enhancer_b.enhance(1.02)
  return enhanced.filter(
      ImageFilter.UnsharpMask(radius=1.2, percent=120, threshold=2)
  )


# === 3. 3枚目専用：側面（背景純白化） ===
def enhance_side(img: Image.Image) -> Image.Image:
  img.thumbnail((1200, 1200), Image.Resampling.LANCZOS)
  img = img.convert("RGB")
  dark_mode = is_dark_item(img)

  lut = []
  for i in range(256):
    x = i / 255.0
    if dark_mode:
      if x < 0.45:
        y = math.pow(x, 0.75) * 1.12
      elif x < 0.70:
        t = (x - 0.45) / (0.70 - 0.45)
        y_45 = math.pow(0.45, 0.75) * 1.12
        y = y_45 + t * (0.95 - y_45)
      else:
        t = (x - 0.70) / 0.30
        y = 0.95 + t * 0.05
    else:
      y = x
    val = min(255, max(0, int(y * 255.0)))
    lut.append(val)

  enhanced = img.point(lut * 3)

  if dark_mode:
    enhancer_b = ImageEnhance.Brightness(enhanced)
    enhanced = enhancer_b.enhance(1.14)
    enhancer_c = ImageEnhance.Contrast(enhanced)
    enhanced = enhancer_c.enhance(1.15)

  return enhanced.filter(
      ImageFilter.UnsharpMask(radius=1.2, percent=120, threshold=2)
  )


# === 4. 4枚目専用：角スレ8箇所（★新設：黒色強化＆背景白保持） ===
def enhance_corner(img: Image.Image) -> Image.Image:
  img.thumbnail((1200, 1200), Image.Resampling.LANCZOS)
  img = img.convert("RGB")
  dark_mode = is_dark_item(img)

  lut = []
  for i in range(256):
    x = i / 255.0
    if dark_mode:
      # 暗部〜中間部の黒さを深め、白飛びを防ぐ
      if x < 0.45:
        y = math.pow(x, 0.88) * 1.02
      elif x < 0.75:
        t = (x - 0.45) / (0.75 - 0.45)
        y_45 = math.pow(0.45, 0.88) * 1.02
        y = y_45 + t * (0.95 - y_45)
      else:
        # 背景の白さはキープ
        t = (x - 0.75) / 0.25
        y = 0.95 + t * 0.05
    else:
      y = x
    val = min(255, max(0, int(y * 255.0)))
    lut.append(val)

  enhanced = img.point(lut * 3)

  if dark_mode:
    # コントラストを＋18%強めて黒スレ・革の深みを強調
    enhancer_c = ImageEnhance.Contrast(enhanced)
    enhanced = enhancer_c.enhance(1.18)
    enhancer_b = ImageEnhance.Brightness(enhanced)
    enhanced = enhancer_b.enhance(1.02)

  return enhanced.filter(
      ImageFilter.UnsharpMask(radius=1.2, percent=120, threshold=2)
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
  cell_w = 1000 // n
  cell_h = 1000
  collage = Image.new("RGB", (1000, 1000), (255, 255, 255))
  for idx, img in enumerate(images):
    collage.paste(crop_center(img, cell_w, cell_h), (idx * cell_w, 0))
  return collage


def make_4x2_corners_grid(images: list[Image.Image]) -> Image.Image:
  if not images:
    return None
  cols, rows = 2, 4
  cell_w, cell_h = 500, 250
  collage = Image.new("RGB", (1000, 1000), (255, 255, 255))
  for idx, img in enumerate(images[:8]):
    r, c = idx // cols, idx % cols
    collage.paste(crop_center(img, cell_w, cell_h), (c * cell_w, r * cell_h))
  return collage


def make_2x2_grid(images: list[Image.Image]) -> Image.Image:
  if not images:
    return None
  cols, rows = 2, 2
  cell_w, cell_h = 500, 500
  collage = Image.new("RGB", (1000, 1000), (255, 255, 255))
  for idx, img in enumerate(images[:4]):
    r, c = idx // cols, idx % cols
    collage.paste(crop_center(img, cell_w, cell_h), (c * cell_w, r * cell_h))
  return collage


def make_stacked_vertical(images: list[Image.Image]) -> Image.Image:
  if not images:
    return None
  n = len(images)
  cell_w, cell_h = 1000, 1000 // n
  collage = Image.new("RGB", (1000, 1000), (255, 255, 255))
  for idx, img in enumerate(images):
    collage.paste(crop_center(img, cell_w, cell_h), (0, idx * cell_h))
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

  cell_w = target_size // cols
  cell_h = target_size // rows

  collage = Image.new("RGB", (target_size, target_size), (255, 255, 255))
  for idx, img in enumerate(images):
    r, c = idx // cols, idx % cols
    collage.paste(crop_center(img, cell_w, cell_h), (c * cell_w, r * cell_h))
  return collage


# === メインAPI ===
@app.post("/process-images")
async def process_images(request: Request):
  enhanced = []
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

    # ★用途に応じた完全個別処理★
    enhanced = []
    for idx, img in enumerate(images):
      if idx in [0, 1]:  # 1・2枚目（表面・裏面）
        enhanced.append(enhance_surface(img))
      elif 2 <= idx <= 5:  # 3枚目（側面4方向）
        enhanced.append(enhance_side(img))
      elif 6 <= idx <= 13:  # 4枚目（角スレ8箇所）➔ 黒色強化モード
        enhanced.append(enhance_corner(img))
      else:  # 5枚目以降 ➔ 大満足の標準パラメータを完全維持
        enhanced.append(enhance_standard(img))

    out = {}

    def get_slice(start, end=None):
      if end is None:
        return enhanced[start:]
      return enhanced[start:end]

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
        out["10_hardware_etc.jpg"] = make_smart_grid(get_slice(19))

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
        out["10_hardware_etc.jpg"] = make_smart_grid(get_slice(20))

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
        out["10_hardware_etc.jpg"] = make_smart_grid(get_slice(22))

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
        out["10_hardware_etc.jpg"] = make_smart_grid(get_slice(16))

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
        out["10_hardware_etc.jpg"] = make_smart_grid(get_slice(20))

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
