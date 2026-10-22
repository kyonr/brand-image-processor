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

MARGIN = 6
GAP = 6

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

def make_white_boost_lut(gamma: float, boost_strength: float = 0.25) -> list[int]:
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

# === 1. 表面・裏面用補正 ===
def enhance_surface(img: Image.Image) -> Image.Image:
    img.thumbnail((1000, 1000), Image.Resampling.LANCZOS)
    img = img.convert("RGB")
    color_cat = get_item_color_category(img)
    if color_cat == "black":
        enhanced = img.point(make_white_boost_lut(gamma=0.92, boost_strength=0.15))
        enhanced = ImageEnhance.Contrast(enhanced).enhance(1.18)
        enhanced = ImageEnhance.Brightness(enhanced).enhance(1.05)
    elif color_cat == "brown":
        # ★ブラウン強化：ガンマ0.62、明るさ+20%、白抜け強化、彩度微調整
        enhanced = img.point(make_white_boost_lut(gamma=0.62, boost_strength=0.28))
        enhanced = ImageEnhance.Brightness(enhanced).enhance(1.20)
        enhanced = ImageEnhance.Contrast(enhanced).enhance(1.05)
        enhanced = ImageEnhance.Color(enhanced).enhance(1.10)
    else:
        enhanced = img.point(make_white_boost_lut(gamma=0.58, boost_strength=0.28))
        enhanced = ImageEnhance.Brightness(enhanced).enhance(1.12)
        enhanced = ImageEnhance.Contrast(enhanced).enhance(1.04)
    return enhanced.filter(ImageFilter.UnsharpMask(radius=1.2, percent=120, threshold=2))

# === 2. 側面用補正 ===
def enhance_side(img: Image.Image) -> Image.Image:
    img.thumbnail((1000, 1000), Image.Resampling.LANCZOS)
    img = img.convert("RGB")
    color_cat = get_item_color_category(img)
    if color_cat == "black":
        enhanced = img.point(make_white_boost_lut(gamma=0.82, boost_strength=0.15))
        enhanced = ImageEnhance.Brightness(enhanced).enhance(1.14)
        enhanced = ImageEnhance.Contrast(enhanced).enhance(1.15)
    elif color_cat == "brown":
        # ★ブラウン強化：明るさ+20%、彩度+20%で鮮やかさを引き上げ
        enhanced = img.point(make_white_boost_lut(gamma=0.58, boost_strength=0.25))
        enhanced = ImageEnhance.Brightness(enhanced).enhance(1.20)
        enhanced = ImageEnhance.Color(enhanced).enhance(1.20)
    else:
        enhanced = img.point(make_white_boost_lut(gamma=0.62, boost_strength=0.22))
        enhanced = ImageEnhance.Brightness(enhanced).enhance(1.06)
    return enhanced.filter(ImageFilter.UnsharpMask(radius=1.2, percent=120, threshold=2))

# === 3. 角スレ用補正 ===
def enhance_corner(img: Image.Image) -> Image.Image:
    img.thumbnail((1000, 1000), Image.Resampling.LANCZOS)
    img = img.convert("RGB")
    color_cat = get_item_color_category(img)
    if color_cat == "black":
        enhanced = img.point(make_white_boost_lut(gamma=0.92, boost_strength=0.15))
        enhanced = ImageEnhance.Contrast(enhanced).enhance(1.18)
    elif color_cat == "brown":
        # ★ブラウン強化：明るさ+20%、彩度+20%
        enhanced = img.point(make_white_boost_lut(gamma=0.58, boost_strength=0.25))
        enhanced = ImageEnhance.Brightness(enhanced).enhance(1.20)
        enhanced = ImageEnhance.Color(enhanced).enhance(1.20)
    else:
        enhanced = img.point(make_white_boost_lut(gamma=0.62, boost_strength=0.22))
        enhanced = ImageEnhance.Brightness(enhanced).enhance(1.06)
    return enhanced.filter(ImageFilter.UnsharpMask(radius=1.2, percent=120, threshold=2))

# === 4. 標準補正（内側・ポケット用） ===
def enhance_standard(img: Image.Image) -> Image.Image:
    img.thumbnail((1000, 1000), Image.Resampling.LANCZOS)
    img = img.convert("RGB")
    color_cat = get_item_color_category(img)
    if color_cat == "black":
        enhanced = img.point(make_white_boost_lut(gamma=0.80, boost_strength=0.15))
        enhanced = ImageEnhance.Brightness(enhanced).enhance(1.05)
        enhanced = ImageEnhance.Contrast(enhanced).enhance(1.05)
    elif color_cat == "brown":
        # ★ブラウン強化：明るさ+20%で内側の暗さを大幅改善
        enhanced = img.point(make_white_boost_lut(gamma=0.55, boost_strength=0.25))
        enhanced = ImageEnhance.Brightness(enhanced).enhance(1.20)
        enhanced = ImageEnhance.Contrast(enhanced).enhance(1.04)
        enhanced = ImageEnhance.Color(enhanced).enhance(1.12)
    else:
        enhanced = img.point(make_white_boost_lut(gamma=0.60, boost_strength=0.22))
        enhanced = ImageEnhance.Brightness(enhanced).enhance(1.06)
        enhanced = ImageEnhance.Contrast(enhanced).enhance(1.04)
    return enhanced.filter(ImageFilter.UnsharpMask(radius=1.2, percent=120, threshold=2))

# === 5. 金具・刻印・シリアル用 ===
def enhance_hardware(img: Image.Image) -> Image.Image:
    img.thumbnail((1000, 1000), Image.Resampling.LANCZOS)
    img = img.convert("RGB")
    enhanced = ImageEnhance.Brightness(img).enhance(1.15)
    enhanced = ImageEnhance.Contrast(enhanced).enhance(1.05)
    return enhanced.filter(ImageFilter.UnsharpMask(radius=1.0, percent=110, threshold=1))

def crop_center(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    return ImageOps.fit(img, (target_w, target_h), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))

def make_1x4_side_strip(images: list[Image.Image]) -> Image.Image:
    if not images: return None
    n = len(images)
    cell_w = (1000 - MARGIN * 2 - GAP * (n - 1)) // n
    cell_h = 1000 - MARGIN * 2
    collage = Image.new("RGB", (1000, 1000), (255, 255, 255))
    for idx, img in enumerate(images):
        collage.paste(crop_center(img, cell_w, cell_h), (MARGIN + idx * (cell_w + GAP), MARGIN))
    return collage

def make_4x2_corners_grid(images: list[Image.Image]) -> Image.Image:
    if not images: return None
    cols, rows = 2, 4
    cell_w = (1000 - MARGIN * 2 - GAP * (cols - 1)) // cols
    cell_h = (1000 - MARGIN * 2 - GAP * (rows - 1)) // rows
    collage = Image.new("RGB", (1000, 1000), (255, 255, 255))
    for idx, img in enumerate(images[:8]):
        r, c = idx // cols, idx % cols
        collage.paste(crop_center(img, cell_w, cell_h), (MARGIN + c * (cell_w + GAP), MARGIN + r * (cell_h + GAP)))
    return collage

def make_2x2_grid(images: list[Image.Image]) -> Image.Image:
    if not images: return None
    cols, rows = 2, 2
    cell_w = (1000 - MARGIN * 2 - GAP * (cols - 1)) // cols
    cell_h = (1000 - MARGIN * 2 - GAP * (rows - 1)) // rows
    collage = Image.new("RGB", (1000, 1000), (255, 255, 255))
    for idx, img in enumerate(images[:4]):
        r, c = idx // cols, idx % cols
        collage.paste(crop_center(img, cell_w, cell_h), (MARGIN + c * (cell_w + GAP), MARGIN + r * (cell_h + GAP)))
    return collage

def make_stacked_vertical(images: list[Image.Image]) -> Image.Image:
    if not images: return None
    n = len(images)
    cell_w = 1000 - MARGIN * 2
    cell_h = (1000 - MARGIN * 2 - GAP * (n - 1)) // n
    collage = Image.new("RGB", (1000, 1000), (255, 255, 255))
    for idx, img in enumerate(images):
        collage.paste(crop_center(img, cell_w, cell_h), (MARGIN, MARGIN + idx * (cell_h + GAP)))
    return collage

def make_smart_grid(images: list[Image.Image], target_size: int = 1000) -> Image.Image:
    if not images: return None
    n = len(images)
    if n == 1: return crop_center(images[0], target_size, target_size)
    if n <= 2: cols, rows = 2, 1
    elif n <= 4: cols, rows = 2, 2
    elif n <= 6: cols, rows = 3, 2
    elif n <= 9: cols, rows = 3, 3
    else: cols, rows = 4, (n + 3) // 4
    cell_w = (target_size - MARGIN * 2 - GAP * (cols - 1)) // cols
    cell_h = (target_size - MARGIN * 2 - GAP * (rows - 1)) // rows
    collage = Image.new("RGB", (target_size, target_size), (255, 255, 255))
    for idx, img in enumerate(images):
        r, c = idx // cols, idx % cols
        collage.paste(crop_center(img, cell_w, cell_h), (MARGIN + c * (cell_w + GAP), MARGIN + r * (cell_h + GAP)))
    return collage

@app.post("/process-images")
async def process_images(request: Request):
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
            if idx in [0, 1]:
                enhanced.append(enhance_surface(img))
            elif 2 <= idx <= 5:
                enhanced.append(enhance_side(img))
            elif 6 <= idx <= 13:
                enhanced.append(enhance_corner(img))
            else:
                enhanced.append(enhance_standard(img))
            
            if idx >= 15:
                enhanced_hw.append(enhance_hardware(img))

        images.clear()
        gc.collect()

        out = {}

        def get_slice(start, end=None, hw=False):
            source = enhanced_hw if hw else enhanced
            if hw:
                hw_start = max(0, start - 15)
                hw_end = None if end is None else max(0, end - 15)
                return source[hw_start:hw_end]
            if end is None:
                return source[start:]
            return source[start:end]

        if p_type == "wallet_small":
            if len(enhanced) >= 1: out["01_surface.jpg"] = crop_center(enhanced[0], 1000, 1000)
            if len(enhanced) >= 2: out["02_back.jpg"] = crop_center(enhanced[1], 1000, 1000)
            if len(enhanced) >= 6: out["03_sides_1x4.jpg"] = make_1x4_side_strip(get_slice(2, 6))
            if len(enhanced) >= 14: out["04_corners_4x2.jpg"] = make_4x2_corners_grid(get_slice(6, 14))
            if len(enhanced) >= 15: out["05_inside.jpg"] = crop_center(enhanced[14], 1000, 1000)
            if len(enhanced) >= 16: out["06_pocket1.jpg"] = crop_center(enhanced[15], 1000, 1000)
            if len(enhanced) >= 17: out["07_pocket2.jpg"] = crop_center(enhanced[16], 1000, 1000)
            if len(enhanced) >= 18: out["08_pocket3.jpg"] = crop_center(enhanced[17], 1000, 1000)
            if len(enhanced) >= 19: out["09_accessories.jpg"] = crop_center(enhanced[18], 1000, 1000)
            if len(enhanced) >= 20: out["10_hardware_etc.jpg"] = make_smart_grid(get_slice(19, hw=True))

        elif p_type == "wallet_medium":
            if len(enhanced) >= 1: out["01_surface.jpg"] = crop_center(enhanced[0], 1000, 1000)
            if len(enhanced) >= 2: out["02_back.jpg"] = crop_center(enhanced[1], 1000, 1000)
            if len(enhanced) >= 6: out["03_sides_1x4.jpg"] = make_1x4_side_strip(get_slice(2, 6))
            if len(enhanced) >= 14: out["04_corners_4x2.jpg"] = make_4x2_corners_grid(get_slice(6, 14))
            if len(enhanced) >= 15: out["05_inside.jpg"] = crop_center(enhanced[14], 1000, 1000)
            if len(enhanced) >= 17: out["06_pockets_2x1.jpg"] = make_stacked_vertical(get_slice(15, 17))
            if len(enhanced) >= 18: out["07_pocket1.jpg"] = crop_center(enhanced[17], 1000, 1000)
            if len(enhanced) >= 19: out["08_pocket2.jpg"] = crop_center(enhanced[18], 1000, 1000)
            if len(enhanced) >= 20: out["09_accessories.jpg"] = crop_center(enhanced[19], 1000, 1000)
            if len(enhanced) >= 21: out["10_hardware_etc.jpg"] = make_smart_grid(get_slice(20, hw=True))

        elif p_type == "wallet_large":
            if len(enhanced) >= 1: out["01_surface.jpg"] = crop_center(enhanced[0], 1000, 1000)
            if len(enhanced) >= 2: out["02_back.jpg"] = crop_center(enhanced[1], 1000, 1000)
            if len(enhanced) >= 6: out["03_sides_1x4.jpg"] = make_1x4_side_strip(get_slice(2, 6))
            if len(enhanced) >= 14: out["04_corners_4x2.jpg"] = make_4x2_corners_grid(get_slice(6, 14))
            if len(enhanced) >= 15: out["05_inside.jpg"] = crop_center(enhanced[14], 1000, 1000)
            if len(enhanced) >= 19: out["06_pockets_2x2.jpg"] = make_2x2_grid(get_slice(15, 19))
            if len(enhanced) >= 20: out["07_pocket1.jpg"] = crop_center(enhanced[19], 1000, 1000)
            if len(enhanced) >= 21: out["08_pocket2.jpg"] = crop_center(enhanced[20], 1000, 1000)
            if len(enhanced) >= 22: out["09_accessories.jpg"] = crop_center(enhanced[21], 1000, 1000)
            if len(enhanced) >= 23: out["10_hardware_etc.jpg"] = make_smart_grid(get_slice(22, hw=True))

        elif p_type == "bag_small":
            if len(enhanced) >= 1: out["01_thumb.jpg"] = crop_center(enhanced[0], 1000, 1000)
            if len(enhanced) >= 2: out["02_front.jpg"] = crop_center(enhanced[1], 1000, 1000)
            if len(enhanced) >= 3: out["03_back.jpg"] = crop_center(enhanced[2], 1000, 1000)
            if len(enhanced) >= 5: out["04_sides_1x2.jpg"] = make_1x4_side_strip(get_slice(3
