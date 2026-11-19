import base64
import gc
import io
import os
import re
import zipfile
from fastapi import FastAPI, HTTPException, Request, Response
from google import genai
from google.genai import types
from PIL import Image, ImageOps
from pydantic import BaseModel, Field
import requests

app = FastAPI()


class ListingOutput(BaseModel):
    title: str = Field(description="メルカリ用商品タイトル（必ず40文字以内）")
    description: str = Field(
        description="メルカリ用商品説明文（指定フォーマット通り。必ず950文字以内。サイズ項目は必須。生産国は記載しない）"
    )


@app.get("/")
def read_root():
    return {"status": "Brand Image & Listing Processor API is running"}


# ==========================================
# 1. 画像結合用レイアウト関数（/process-images）
# ==========================================
MARGIN = 6
GAP = 6


def crop_center(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    return ImageOps.fit(
        img, (target_w, target_h), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5)
    )


def make_1x4_side_strip(images: list[Image.Image]) -> Image.Image:
    if not images:
        return None
    n = len(images)
    cell_w = (1000 - MARGIN * 2 - GAP * (n - 1)) // n
    cell_h = 1000 - MARGIN * 2
    collage = Image.new("RGB", (1000, 1000), (255, 255, 255))
    for idx, img in enumerate(images):
        collage.paste(
            crop_center(img, cell_w, cell_h), (MARGIN + idx * (cell_w + GAP), MARGIN)
        )
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
        collage.paste(
            crop_center(img, cell_w, cell_h),
            (MARGIN + c * (cell_w + GAP), MARGIN + r * (cell_h + GAP)),
        )
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
        collage.paste(
            crop_center(img, cell_w, cell_h),
            (MARGIN + c * (cell_w + GAP), MARGIN + r * (cell_h + GAP)),
        )
    return collage


def make_stacked_vertical(images: list[Image.Image]) -> Image.Image:
    if not images:
        return None
    n = len(images)
    cell_w = 1000 - MARGIN * 2
    cell_h = (1000 - MARGIN * 2 - GAP * (n - 1)) // n
    collage = Image.new("RGB", (1000, 1000), (255, 255, 255))
    for idx, img in enumerate(images):
        collage.paste(
            crop_center(img, cell_w, cell_h), (MARGIN, MARGIN + idx * (cell_h + GAP))
        )
    return collage


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
        collage.paste(
            crop_center(images[1], cell_w, cell_h), (MARGIN + cell_w + GAP, MARGIN)
        )
    elif n == 3:
        h_half = (target_size - MARGIN * 2 - GAP) // 2
        w_top = target_size - MARGIN * 2
        collage.paste(crop_center(images[0], w_top, h_half), (MARGIN, MARGIN))
        w_bottom = (target_size - MARGIN * 2 - GAP) // 2
        y_bottom = MARGIN + h_half + GAP
        collage.paste(crop_center(images[1], w_bottom, h_half), (MARGIN, y_bottom))
        collage.paste(
            crop_center(images[2], w_bottom, h_half),
            (MARGIN + w_bottom + GAP, y_bottom),
        )
    elif n == 4:
        return make_2x2_grid(images)
    elif n == 5:
        h_half = (target_size - MARGIN * 2 - GAP) // 2
        w_top = (target_size - MARGIN * 2 - GAP) // 2
        for i in range(2):
            collage.paste(
                crop_center(images[i], w_top, h_half),
                (MARGIN + i * (w_top + GAP), MARGIN),
            )
        w_bottom = (target_size - MARGIN * 2 - GAP * 2) // 3
        y_bottom = MARGIN + h_half + GAP
        for i in range(3):
            collage.paste(
                crop_center(images[2 + i], w_bottom, h_half),
                (MARGIN + i * (w_bottom + GAP), y_bottom),
            )
    elif n == 6:
        cols, rows = 3, 2
        cell_w = (target_size - MARGIN * 2 - GAP * (cols - 1)) // cols
        cell_h = (target_size - MARGIN * 2 - GAP * (rows - 1)) // rows
        for idx, img in enumerate(images[:6]):
            r, c = idx // cols, idx % cols
            collage.paste(
                crop_center(img, cell_w, cell_h),
                (MARGIN + c * (cell_w + GAP), MARGIN + r * (cell_h + GAP)),
            )
    elif n == 7:
        h_half = (target_size - MARGIN * 2 - GAP) // 2
        w_top = (target_size - MARGIN * 2 - GAP * 2) // 3
        for i in range(3):
            collage.paste(
                crop_center(images[i], w_top, h_half),
                (MARGIN + i * (w_top + GAP), MARGIN),
            )
        w_bottom = (target_size - MARGIN * 2 - GAP * 3) // 4
        y_bottom = MARGIN + h_half + GAP
        for i in range(4):
            collage.paste(
                crop_center(images[3 + i], w_bottom, h_half),
                (MARGIN + i * (w_bottom + GAP), y_bottom),
            )
    elif n == 8:
        cols, rows = 4, 2
        cell_w = (target_size - MARGIN * 2 - GAP * (cols - 1)) // cols
        cell_h = (target_size - MARGIN * 2 - GAP * (rows - 1)) // rows
        for idx, img in enumerate(images[:8]):
            r, c = idx // cols, idx % cols
            collage.paste(
                crop_center(img, cell_w, cell_h),
                (MARGIN + c * (cell_w + GAP), MARGIN + r * (cell_h + GAP)),
            )
    else:
        cols, rows = 3, 3
        cell_w = (target_size - MARGIN * 2 - GAP * (cols - 1)) // cols
        cell_h = (target_size - MARGIN * 2 - GAP * (rows - 1)) // rows
        for idx, img in enumerate(images[:9]):
            r, c = idx // cols, idx % cols
            collage.paste(
                crop_center(img, cell_w, cell_h),
                (MARGIN + c * (cell_w + GAP), MARGIN + r * (cell_h + GAP)),
            )

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
            if len(images) >= 1:
                out["01_surface.jpg"] = crop_center(images[0], 1000, 1000)
            if len(images) >= 2:
                out["02_back.jpg"] = crop_center(images[1], 1000, 1000)
            if len(images) >= 6:
                out["03_sides_1x4.jpg"] = make_1x4_side_strip(get_slice(2, 6))
            if len(images) >= 14:
                out["04_corners_4x2.jpg"] = make_4x2_corners_grid(get_slice(6, 14))
            if len(images) >= 15:
                out["05_inside.jpg"] = crop_center(images[14], 1000, 1000)
            if len(images) >= 16:
                out["06_pocket1.jpg"] = crop_center(images[15], 1000, 1000)
            if len(images) >= 17:
                out["07_pocket2.jpg"] = crop_center(images[16], 1000, 1000)
            if len(images) >= 18:
                out["08_pocket3.jpg"] = crop_center(images[17], 1000, 1000)
            if len(images) >= 19:
                out["09_accessories.jpg"] = crop_center(images[18], 1000, 1000)
            if len(images) >= 20:
                out["10_hardware_etc.jpg"] = make_smart_grid(get_slice(19))

        elif p_type == "wallet_medium":
            if len(images) >= 1:
                out["01_surface.jpg"] = crop_center(images[0], 1000, 1000)
            if len(images) >= 2:
                out["02_back.jpg"] = crop_center(images[1], 1000, 1000)
            if len(images) >= 6:
                out["03_sides_1x4.jpg"] = make_1x4_side_strip(get_slice(2, 6))
            if len(images) >= 14:
                out["04_corners_4x2.jpg"] = make_4x2_corners_grid(get_slice(6, 14))
            if len(images) >= 15:
                out["05_inside.jpg"] = crop_center(images[14], 1000, 1000)
            if len(images) >= 17:
                out["06_pockets_2x1.jpg"] = make_stacked_vertical(get_slice(15, 17))
            if len(images) >= 18:
                out["07_pocket1.jpg"] = crop_center(images[17], 1000, 1000)
            if len(images) >= 19:
                out["08_pocket2.jpg"] = crop_center(images[18], 1000, 1000)
            if len(images) >= 20:
                out["09_accessories.jpg"] = crop_center(images[19], 1000, 1000)
            if len(images) >= 21:
                out["10_hardware_etc.jpg"] = make_smart_grid(get_slice(20))

        elif p_type == "wallet_large":
            if len(images) >= 1:
                out["01_surface.jpg"] = crop_center(images[0], 1000, 1000)
            if len(images) >= 2:
                out["02_back.jpg"] = crop_center(images[1], 1000, 1000)
            if len(images) >= 6:
                out["03_sides_1x4.jpg"] = make_1x4_side_strip(get_slice(2, 6))
            if len(images) >= 14:
                out["04_corners_4x2.jpg"] = make_4x2_corners_grid(get_slice(6, 14))
            if len(images) >= 15:
                out["05_inside.jpg"] = crop_center(images[14], 1000, 1000)
            if len(images) >= 19:
                out["06_pockets_2x2.jpg"] = make_2x2_grid(get_slice(15, 19))
            if len(images) >= 20:
                out["07_pocket1.jpg"] = crop_center(images[19], 1000, 1000)
            if len(images) >= 21:
                out["08_pocket2.jpg"] = crop_center(images[20], 1000, 1000)
            if len(images) >= 22:
                out["09_accessories.jpg"] = crop_center(images[21], 1000, 1000)
            if len(images) >= 23:
                out["10_hardware_etc.jpg"] = make_smart_grid(get_slice(22))

        elif p_type == "bag_small":
            if len(images) >= 1:
                out["01_thumb.jpg"] = crop_center(images[0], 1000, 1000)
            if len(images) >= 2:
                out["02_front.jpg"] = crop_center(images[1], 1000, 1000)
            if len(images) >= 3:
                out["03_back.jpg"] = crop_center(images[2], 1000, 1000)
            if len(images) >= 5:
                out["04_sides_1x2.jpg"] = make_1x4_side_strip(get_slice(3, 5))
            if len(images) >= 7:
                out["05_top_bottom_2x1.jpg"] = make_stacked_vertical(get_slice(5, 7))
            if len(images) >= 11:
                out["06_corners_2x2.jpg"] = make_2x2_grid(get_slice(7, 11))
            if len(images) >= 13:
                out["07_handles_2x1.jpg"] = make_stacked_vertical(get_slice(11, 13))
            if len(images) >= 14:
                out["08_inside.jpg"] = crop_center(images[13], 1000, 1000)
            if len(images) >= 16:
                out["09_pockets_2x1.jpg"] = make_stacked_vertical(get_slice(14, 16))
            if len(images) >= 17:
                out["10_hardware_etc.jpg"] = make_smart_grid(get_slice(16))

        elif p_type == "bag_large":
            if len(images) >= 1:
                out["01_thumb.jpg"] = crop_center(images[0], 1000, 1000)
            if len(images) >= 2:
                out["02_front.jpg"] = crop_center(images[1], 1000, 1000)
            if len(images) >= 3:
                out["03_back.jpg"] = crop_center(images[2], 1000, 1000)
            if len(images) >= 7:
                out["04_sides_top_bottom_2x2.jpg"] = make_2x2_grid(get_slice(3, 7))
            if len(images) >= 11:
                out["05_corners_2x2.jpg"] = make_2x2_grid(get_slice(7, 11))
            if len(images) >= 13:
                out["06_handles_2x1.jpg"] = make_stacked_vertical(get_slice(11, 13))
            if len(images) >= 14:
                out["07_inside.jpg"] = crop_center(images[13], 1000, 1000)
            if len(images) >= 16:
                out["08_pockets_2x1.jpg"] = make_stacked_vertical(get_slice(14, 16))
            if len(images) >= 20:
                out["09_pockets_2x2.jpg"] = make_2x2_grid(get_slice(16, 20))
            if len(images) >= 21:
                out["10_hardware_etc.jpg"] = make_smart_grid(get_slice(20))

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

        gc.collect()
        zip_buf.seek(0)
        return Response(
            content=zip_buf.getvalue(),
            media_type="application/zip",
            headers={"Content-Disposition": "attachment; filename=processed_images.zip"},
        )

    except Exception as e:
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("error.txt", f"Processing error: {str(e)}")
        zip_buf.seek(0)
        return Response(
            content=zip_buf.getvalue(),
            media_type="application/zip",
            headers={"Content-Disposition": "attachment; filename=processed_images.zip"},
        )


# ==========================================
# 2. メルカリ下書き準備関数（/prepare-listing）
# ==========================================
def sanitize_folder_name(name: str) -> str:
    cleaned = re.sub(r'[\\/:*?"<>|\r\n]+', " ", name).strip()
    return cleaned[:80] if cleaned else "新規商品"


def natural_sort_key(filename: str):
    base = os.path.basename(filename)
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split(r"(\d+)", base)
    ]


@app.post("/prepare-listing")
async def prepare_listing(request: Request):
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    gas_webapp_url = os.environ.get("GAS_WEBAPP_URL")

    if not gemini_api_key:
        raise HTTPException(
            status_code=500, detail="GEMINI_API_KEY が設定されていません"
        )
    if not gas_webapp_url:
        raise HTTPException(
            status_code=500, detail="GAS_WEBAPP_URL が設定されていません"
        )

    body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail="ZIPデータが空です")

    try:
        zip_file = zipfile.ZipFile(io.BytesIO(body))
    except zipfile.BadZipFile:
        raise HTTPException(
            status_code=400, detail="ZIPファイルの形式が正しくありません"
        )

    info_text = ""
    image_entries = []

    for entry in sorted(zip_file.namelist(), key=natural_sort_key):
        base_name = os.path.basename(entry)
        if not base_name or base_name.startswith(".") or "__MACOSX" in entry:
            continue
        lower_name = base_name.lower()
        if lower_name == "info.txt":
            raw_txt = zip_file.read(entry)
            info_text = raw_txt.decode("utf-8-sig", errors="replace").strip()
        elif lower_name.endswith((".jpg", ".jpeg", ".png", ".webp", ".heic")):
            image_entries.append(entry)

    if not info_text:
        raise HTTPException(
            status_code=400, detail="ZIP内に info.txt が見つかりませんでした"
        )
    if not image_entries:
        raise HTTPException(
            status_code=400, detail="ZIP内に画像ファイルが見つかりませんでした"
        )

    image_parts = []
    base64_images = []

    for idx, entry in enumerate(image_entries):
        raw_img = zip_file.read(entry)
        img = Image.open(io.BytesIO(raw_img))
        img = ImageOps.exif_transpose(img)
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.thumbnail((1600, 1600), Image.Resampling.LANCZOS)

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=88)
        jpeg_bytes = buf.getvalue()

        base64_images.append(base64.b64encode(jpeg_bytes).decode("utf-8"))
        if idx < 10:
            image_parts.append(f"【写真{idx + 1}枚目】")
            image_parts.append(
                types.Part.from_bytes(data=jpeg_bytes, mime_type="image/jpeg")
            )

    prompt = f"""あなたは中古ブランド品販売のプロの出品担当者です。
入力情報と商品画像（1枚目〜順番通り）を確認し、メルカリ出品用の「タイトル」と「商品説明文」を作成してください。

【入力情報（管理番号・商品名など）】
{info_text}

【厳守ルール】
1. タイトルは全角40文字以内に必ず収めること（入力情報のキャッチコピーやブランド名・モデル名・カラー等を活かし、検索されやすくまとめる）。
2. 商品説明文は全体で950文字以内に必ず収めること。
3. 商品説明文は下記の【出力テンプレート】の絵文字・改行・空行・文体を完全に再現して作成すること。
4. 「・サイズ　：」の項目は**絶対に削除せず必ず記載すること**。入力情報にサイズの記載がある場合はその数値を記載し、記載がない場合でも、特定されたブランド名・モデル名・型番の公式標準サイズ（実寸目安）から「横幅約〇〇cm × 縦約〇〇cm × マチ約〇〇cm」の形式で必ず算出・記載すること（万が一モデル特定すら困難な場合のみ「横幅約 cm × 縦約 cm × マチ約 cm」と枠を残すこと）。
5. 画像から確認できる範囲で丁寧かつ誠実に記載し、確認できないシリアル番号や付属品は勝手に捏造しないこと。
6. 「生産国（原産国）」の項目は記載しないこと。
7. シリアルナンバーが画像から読み取れる場合は「・シリアルナンバー：〇〇（写真〇枚目参照）」と記載し、画像に写っていない・読み取れない場合は「・シリアルナンバー：」の行ごと削除すること。
8. 付属品が画像に写っている場合は「・付属品　：箱、保存袋(写真〇枚目)」のように何枚目に写っているかも記載し、本体のみの場合は「・付属品　：本体のみ」とすること。
9. 最後の行は「管理番号111」のようにコロン（：）を付けずに入力情報の管理番号を記載すること。

【出力テンプレート】
⭐️鑑定済み(正規品)商品のみのお取り扱い✨
⭐️翌日中以内配送✨
⭐️即購入OK✨


（ブランド名（カタカナ表記）） （モデル名・ライン名） のご紹介です✨
（デザインや素材・金具の美しさなど、商品の魅力を伝える1文）✨
（収納力や使い勝手、どんなシーンにおすすめかを伝える1文）✨


⭐️商品詳細⭐️
・ブランド：（ブランド名英語 カタカナ）
・モデル　：（モデル名・ライン名・型番）
・素材　　：（レザー（本革）、PVCコーティングキャンバスなど画像から分かる素材）
・色　　　：（カラー・金具の色）
・サイズ　：横幅約〇〇cm × 縦約〇〇cm × マチ約〇〇cm
・仕　様　：（開閉方式、札入れ・小銭入れ・カードポケット・フラットポケット数など画像から分かる仕様）
・シリアルナンバー：（読み取れる場合のみ「〇〇（写真〇枚目参照）」と記載。不明なら行ごと削除）
・付属品　：（画像に写っている付属品＋「(写真〇枚目)」、なければ「本体のみ」）


⭐️商品状態⭐️
✅外側
✔︎（画像から分かる外側のコンディションを丁寧かつポジティブに記載）✨
✅内側
✔︎（画像から分かる内側のコンディションを丁寧かつ正直に記載）✨
✅ボタン・金具
✔︎（画像から分かるボタン・ファスナー・金具のコンディションを丁寧に記載）✨


⭐️ご注意点⭐️
・出品に際し、クリーニングとメンテナンスを実施しております。
・真贋鑑定を行う仕入れ先での購入の為、真贋に問題ないと認識しておりますが、万が一偽物が混入の際には、受取評価前でしたら返品送料含め全額返金対応させていただきます！


⭐️最後に⭐️
（ブランド名・モデル名・カラー等の人気や魅力をまとめた売り文句）！早い者勝ちですので、ぜひこの機会に手に入れてください！^_^


管理番号（入力情報にある数字のみ）
"""

    try:
        client = genai.Client(api_key=gemini_api_key)
        response = client.models.generate_content(
            model="gemini-3.8-flash",
            contents=[*image_parts, prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ListingOutput,
                temperature=0.3,
            ),
        )
        result = ListingOutput.model_validate_json(response.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini APIエラー: {str(e)}")

    title = result.title.strip()[:40]
    description = result.description.strip()[:950]
    folder_name = sanitize_folder_name(info_text)

    gas_payload = {
        "folder_name": folder_name,
        "title": title,
        "description": description,
        "images": base64_images,
    }

    try:
        gas_res = requests.post(gas_webapp_url, json=gas_payload, timeout=60)
        gas_res.raise_for_status()
        gas_json = gas_res.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"GAS送信エラー: {str(e)}")

    if gas_json.get("status") != "success":
        raise HTTPException(
            status_code=500,
            detail=f"GAS保存エラー: {gas_json.get('message', '不明なエラー')}",
        )

    gc.collect()
    return {
        "status": "success",
        "folder_name": folder_name,
        "title": title,
        "folderUrl": gas_json.get("folderUrl", ""),
    }
