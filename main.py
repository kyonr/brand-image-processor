import io
import os
import zipfile
import gc
from fastapi import FastAPI, HTTPException, Request, Response
from PIL import Image, ImageOps

app = FastAPI()

@app.get("/")
def read_root():
    return {"status": "Brand Image Processor API is running (Layout Only)"}

MARGIN = 6
GAP = 6

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

# 空白なし可変スマートグリッド
def make_smart_grid(images: list[Image.Image], target_size: int = 1000) -> Image.Image:
    if not images:
        return None
    n = len(images)
    collage = Image.new("RGB", (target_size, target_size), (255, 255, 255))

    if n == 1:
        return crop_center(images[0], target_size, target_size)
    elif n == 2:
        cell_w = (target_size - MARGIN * 2 - GAP) // 2
        cell_h = target_size - MARGIN * 2
        collage.paste(crop_center(images[0], cell_w, cell_h), (MARGIN, MARGIN))
        collage.paste(crop_center(images[1], cell_w, cell_h), (MARGIN + cell_w + GAP, MARGIN))
    elif n == 3:
        h_half = (target_size - MARGIN * 2 - GAP) // 2
        w_top = target_size - MARGIN * 2
        collage.paste(crop_center(images[0], w_top, h_half), (MARGIN, MARGIN))
        w_bottom = (target_size - MARGIN * 2 - GAP) // 2
        y_bottom = MARGIN + h_half + GAP
        collage.paste(crop_center(images[1], w_bottom, h_half), (MARGIN, y_bottom))
        collage.paste(crop_center(images[2], w_bottom, h_half), (MARGIN + w_bottom + GAP, y_bottom))
    elif n == 4:
        return make_2x2_grid(images)
    elif n == 5:
        h_half = (target_size - MARGIN * 2 - GAP) // 2
        w_top = (target_size - MARGIN * 2 - GAP) // 2
        for i in range(2):
            collage.paste(crop_center(images[i], w_top, h_half), (MARGIN + i * (w_top + GAP), MARGIN))
        w_bottom = (target_size - MARGIN * 2 - GAP * 2) // 3
        y_bottom = MARGIN + h_half + GAP
        for i in range(3):
            collage.paste(crop_center(images[2 + i], w_bottom, h_half), (MARGIN + i * (w_bottom + GAP), y_bottom))
    elif n == 6:
        cols, rows = 3, 2
        cell_w = (target_size - MARGIN * 2 - GAP * (cols - 1)) // cols
        cell_h = (target_size - MARGIN * 2 - GAP * (rows - 1)) // rows
        for idx, img in enumerate(images[:6]):
            r, c = idx // cols, idx % cols
            collage.paste(crop_center(img, cell_w, cell_h), (MARGIN + c * (cell_w + GAP), MARGIN + r * (cell_h + GAP)))
    elif n == 7:
        h_half = (target_size - MARGIN * 2 - GAP) // 2
        w_top = (target_size - MARGIN * 2 - GAP * 2) // 3
        for i in range(3):
            collage.paste(crop_center(images[i], w_top, h_half), (MARGIN + i * (w_top + GAP), MARGIN))
        w_bottom = (target_size - MARGIN * 2 - GAP * 3) // 4
        y_bottom = MARGIN + h_half + GAP
        for i in range(4):
            collage.paste(crop_center(images[3 + i], w_bottom, h_half), (MARGIN + i * (w_bottom + GAP), y_bottom))
    elif n == 8:
        cols, rows = 4, 2
        cell_w = (target_size - MARGIN * 2 - GAP * (cols - 1)) // cols
        cell_h = (target_size - MARGIN * 2 - GAP * (rows - 1)) // rows
        for idx, img in enumerate(images[:8]):
            r, c = idx // cols, idx % cols
            collage.paste(crop_center(img, cell_w, cell_h), (MARGIN + c * (cell_w + GAP), MARGIN + r * (cell_h + GAP)))
    else:
        cols, rows = 3, 3
        cell_w = (target_size - MARGIN * 2 - GAP * (cols - 1)) // cols
        cell_h = (target_size - MARGIN * 2 - GAP * (rows - 1)) // rows
        for idx, img in enumerate(images[:9]):
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
                    img = Image.open(io.BytesIO(z.read(filename))).convert("RGB")
                    images.append(img)

        if not images:
            raise HTTPException(status_code=400, detail="No images")

        out = {}

        def get_slice(start, end=None):
            if end is None:
                return images[start:]
            return images[start:end]

        if p_type == "wallet_small":
            if len(images) >= 1: out["01_surface.jpg"] = crop_center(images[0], 1000, 1000)
            if len(images) >= 2: out["02_back.jpg"] = crop_center(images[1], 1000, 1000)
            if len(images) >= 6: out["03_sides_1x4.jpg"] = make_1x4_side_strip(get_slice(2, 6))
            if len(images) >= 14: out["04_corners_4x2.jpg"] = make_4x2_corners_grid(get_slice(6, 14))
            if len(images) >= 15: out["05_inside.jpg"] = crop_center(images[14], 1000, 1000)
            if len(images) >= 16: out["06_pocket1.jpg"] = crop_center(images[15], 1000, 1000)
            if len(images) >= 17: out["07_pocket2.jpg"] = crop_center(images[16], 1000, 1000)
            if len(images) >= 18: out["08_pocket3.jpg"] = crop_center(images[17], 1000, 1000)
            if len(images) >= 19: out["09_accessories.jpg"] = crop_center(images[18], 1000, 1000)
            if len(images) >= 20: out["10_hardware_etc.jpg"] = make_smart_grid(get_slice(19))

        elif p_type == "wallet_medium":
            if len(images) >= 1: out["01_surface.jpg"] = crop_center(images[0], 1000, 1000)
            if len(images) >= 2: out["02_back.jpg"] = crop_center(images[1], 1000, 1000)
            if len(images) >= 6: out["03_sides_1x4.jpg"] = make_1x4_side_strip(get_slice(2, 6))
            if len(images) >= 14: out["04_corners_4x2.jpg"] = make_4x2_corners_grid(get_slice(6, 14))
            if len(images) >= 15: out["05_inside.jpg"] = crop_center(images[14], 1000, 1000)
            if len(images) >= 17: out["06_pockets_2x1.jpg"] = make_stacked_vertical(get_slice(15, 17))
            if len(images) >= 18: out["07_pocket1.jpg"] = crop_center(images[17], 1000, 1000)
            if len(images) >= 19: out["08_pocket2.jpg"] = crop_center(images[18], 1000, 1000)
            if len(images) >= 20: out["09_accessories.jpg"] = crop_center(images[19], 1000, 1000)
            if len(images) >= 21: out["10_hardware_etc.jpg"] = make_smart_grid(get_slice(20))

        elif p_type == "wallet_large":
            if len(images) >= 1: out["01_surface.jpg"] = crop_center(images[0], 1000, 1000)
            if len(images) >= 2: out["02_back.jpg"] = crop_center(images[1], 1000, 1000)
            if len(images) >= 6: out["03_sides_1x4.jpg"] = make_1x4_side_strip(get_slice(2, 6))
            if len(images) >= 14: out["04_corners_4x2.jpg"] = make_4x2_corners_grid(get_slice(6, 14))
            if len(images) >= 15: out["05_inside.jpg"] = crop_center(images[14], 1000, 1000)
            if len(images) >= 19: out["06_pockets_2x2.jpg"] = make_2x2_grid(get_slice(15, 19))
            if len(images) >= 20: out["07_pocket1.jpg"] = crop_center(images[19], 1000, 1000)
            if len(images) >= 21: out["08_pocket2.jpg"] = crop_center(images[20], 1000, 1000)
            if len(images) >= 22: out["09_accessories.jpg"] = crop_center(images[21], 1000, 1000)
            if len(images) >= 23: out["10_hardware_etc.jpg"] = make_smart_grid(get_slice(22))

        elif p_type == "bag_small":
            if len(images) >= 1: out["01_thumb.jpg"] = crop_center(images[0], 1000, 1000)
            if len(images) >= 2: out["02_front.jpg"] = crop_center(images[1], 1000, 1000)
            if len(images) >= 3: out["03_back.jpg"] = crop_center(images[2], 1000, 1000)
            if len(images) >= 5: out["04_sides_1x2.jpg"] = make_1x4_side_strip(get_slice(3, 5))
            if len(images) >= 7: out["05_top_bottom_2x1.jpg"] = make_stacked_vertical(get_slice(5, 7))
            if len(images) >= 11: out["06_corners_2x2.jpg"] = make_2x2_grid(get_slice(7, 11))
            if len(images) >= 13: out["07_handles_2x1.jpg"] = make_stacked_vertical(get_slice(11, 13))
            if len(images) >= 14: out["08_inside.jpg"] = crop_center(images[13], 1000, 1000)
            if len(images) >= 16: out["09_pockets_2x1.jpg"] = make_stacked_vertical(get_slice(14, 16))
            if len(images) >= 17: out["10_hardware_etc.jpg"] = make_smart_grid(get_slice(16))

        elif p_type == "bag_large":
            if len(images) >= 1: out["01_thumb.jpg"] = crop_center(images[0], 1000, 1000)
            if len(images) >= 2: out["02_front.jpg"] = crop_center(images[1], 1000, 1000)
            if len(images) >= 3: out["03_back.jpg"] = crop_center(images[2], 1000, 1000)
            if len(images) >= 7: out["04_sides_top_bottom_2x2.jpg"] = make_2x2_grid(get_slice(3, 7))
            if len(images) >= 11: out["05_corners_2x2.jpg"] = make_2x2_grid(get_slice(7, 11))
            if len(images) >= 13: out["06_handles_2x1.jpg"] = make_stacked_vertical(get_slice(11, 13))
            if len(images) >= 14: out["07_inside.jpg"] = crop_center(images[13], 1000, 1000)
            if len(images) >= 16: out["08_pockets_2x1.jpg"] = make_stacked_vertical(get_slice(14, 16))
            if len(images) >= 20: out["09_pockets_2x2.jpg"] = make_2x2_grid(get_slice(16, 20))
            if len(images) >= 21: out["10_hardware_etc.jpg"] = make_smart_grid(get_slice(20))

        else:
            for idx, img in enumerate(images):
                out[f"{idx+1:02d}_photo.jpg"] = crop_center(img, 1000, 1000)

        if not out:
            for idx, img in enumerate(images):
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
            headers={"Content-Disposition": "attachment; filename=processed_images.zip"}
        )

    except Exception as e:
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("error.txt", f"Processing error: {str(e)}")
        zip_buf.seek(0)
        return Response(
            content=zip_buf.getvalue(),
            media_type="application/zip",
            headers={"Content-Disposition": "attachment; filename=processed_images.zip"}
        )
