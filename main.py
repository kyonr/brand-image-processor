import io
import os
import re
import json
import base64
import zipfile
import gc
import requests
from fastapi import FastAPI, HTTPException, Request, Response
from PIL import Image, ImageOps
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

app = FastAPI()

@app.get("/")
def read_root():
    return {"status": "Brand Image & Listing Processor API is running"}

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

# ==========================================
# ① 既存の画像結合用エンドポイント（そのまま維持）
# ==========================================
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

# ==========================================
# ② 新しく追加する「文章生成＆Drive保存」エンドポイント
# ==========================================
class MercariListing(BaseModel):
    title: str = Field(description="メルカリ用の商品タイトル（全角・半角問わず必ず40文字以内・34〜40文字が理想）")
    description: str = Field(description="メルカリ用の商品説明文（絵文字・スペース・改行・管理番号を含め全体で必ず850文字以内・700〜820文字がベスト）")

@app.post("/prepare-listing")
async def prepare_listing(request: Request):
    try:
        body = await request.body()
        if not body:
            raise HTTPException(status_code=400, detail="No data")

        item_text = ""
        images = []

        # 1. ZIPからテキスト(info.txt)と画像(最大20枚)を取り出す
        with zipfile.ZipFile(io.BytesIO(body)) as z:
            for filename in sorted(z.namelist()):
                if filename.startswith("__MACOSX") or filename.startswith("."):
                    continue
                if filename.endswith(".txt"):
                    item_text = z.read(filename).decode("utf-8").strip()
                elif filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                    img = Image.open(io.BytesIO(z.read(filename))).convert("RGB")
                    images.append(img)

        if not images:
            raise HTTPException(status_code=400, detail="No images found in ZIP")

        target_images = images[:20]

        # 2. Gemini APIで指定テンプレート通りにタイトルと説明文を生成
        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

        prompt = f"""
あなたは中古ブランド品リペア・販売のプロの出品アシスタントです。
入力された「商品画像（1枚目〜{len(target_images)}枚目）」および「基本情報（ブランド・モデル・状態ランク・管理番号など）」をもとに、以下の厳格なルールに従って【メルカリ用タイトル】と【商品説明文】を作成してください。

【基本情報】
{item_text}

---

### 1. メルカリ用タイトル作成ルール
- **文字数制限**：**全角・半角問わず必ず40文字以内**に収めること（34〜40文字が理想）。
- **構成フォーマット**：`☆状態ランク☆ブランド名 アイテム名 モデル・ライン名 カラーや特徴`
- **作成のポイント**：
  * 冒頭に必ず `☆極美品☆` `☆未使用に近い☆` `☆箱付極美品☆` `☆箱付美品☆` などの状態ランクを入れる。
  * 句読点や不要な記号は使わず、検索されやすいSEOキーワード（ブランド名、アイテム種別、ライン名、モデル名、色など）をスペース区切りで詰め込む。
- **出力例**：
  * `☆極美品☆グッチ ショルダーバッグ オフィディア GGスプリーム シェリーライン`（40文字）
  * `☆未使用に近い☆セリーヌ 折財布 ファインストラップウォレット トリオンフ`（37文字）
  * `☆箱付極美品☆ロエベ アナグラム トライフォールドウォレット ピンクベージュ`（38文字）

---

### 2. 商品説明文作成ルール
- **文字数制限**：絵文字・スペース・改行・管理番号を含め、**全体で必ず850文字以内**（700〜820文字程度がベスト）に収めること。
- **項目・記載の禁止ルール（厳守）**：
  * **商品詳細欄に「素材」の項目（例：`・素 材 ：レザー`）は作らないこと**（※冒頭の商品紹介文や状態説明の文章中で「レザー」などのワードを自然に使用するのはOK）。
  * **「原産国」は一切記載しないこと**（商品詳細欄の `・原産国 ：イタリア` などの項目はもちろん、本文中でも `Made in Italy / Spain` 等の記載はしない）。
- **画像解析と反映**：
  * 添付された画像（各画像に付与された「写真〇枚目」の番号）から「シリアルナンバー」「付属品の有無」「角スレ・汚れ・金具の傷などのコンディション」を正確に読み取り、該当する写真番号（例：`（写真9枚目右下参照）`）を記載すること。
  * ダメージがある場合は隠さず誠実に記載しつつ、「ご使用時には目立たない程度です✨」「ご使用上全く問題ございません✨」と安心感を与える表現でフォローすること。
- **冒頭紹介文のルール**：
  * 最初の挨拶後の商品紹介3行は、合計100文字以内を目安に簡潔かつ魅力的にまとめること。

---

### 3. 商品説明文の固定テンプレート（厳守）

以下の改行・絵文字・見出しのレイアウトを完全に再現して出力してください。

⭐️鑑定済み(正規品)商品のみのお取り扱い✨
⭐️翌日中以内配送✨
⭐️即購入OK✨


[ブランド名]の[モデル名]のご紹介です✨
[デザインや色・素材感などの魅力を1行で簡潔に]✨
[使い勝手や状態の魅力を1行で簡潔に]✨


⭐️商品詳細⭐️
・ブランド：[英語表記] [カタカナ表記]
・型 番 ：[型番がある場合のみ記載／なければ行ごと削除]
・モデル ：[モデル名]
・ライン ：[ライン名がある場合のみ記載／なければ行ごと削除]
・ 色   ：[メインカラー] × [金具の色]
・サイズ ：横幅約〇cm × 縦約〇cm × マチ約〇cm（※バッグの場合はショルダー長さ等も記載）
・仕 様 ：[開閉方式、小銭入れ・札入れ・カードポケット・フリーポケット数など]
・シリアル：[画像から読み取ったシリアルナンバー]（写真〇枚目参照）
・付属品 ：[なし（本体のみ） または 箱、保存袋など（写真〇枚目参照）]


⭐️商品状態⭐️
✅外側
✔︎ [外側の型崩れ・角擦れ・表面の状態などを記載]✨
✅内側
✔︎ [お札入れ・小銭入れ・ポケット内の汚れや使用感の有無を記載]✨
✅金具・ファスナー（※アイテムに合わせて「✅金具」等に調整）
✔︎ [ロゴ金具の輝きや、ファスナー・スナップボタンの動作状態を記載]✨


⭐️ご注意点⭐️
・出品に際し、クリーニングとメンテナンスを実施しております。
・万が一偽物の場合は受取評価前に限り返品・返金対応させていただきます！


⭐️最後に⭐️
[商品の魅力の再アピールと購入を促す親しみやすい締めくくり]！^_^


管理番号[指定された管理番号]
"""

        # Geminiに「写真1枚目」「写真2枚目」と番号を明示しながら画像を渡す
        gemini_contents = [prompt]
        for idx, img in enumerate(target_images):
            gemini_contents.append(f"--- 写真{idx + 1}枚目 ---")
            gemini_contents.append(img)

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=gemini_contents,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=MercariListing,
                temperature=0.3,
            ),
        )

        result_data = json.loads(response.text)
        title = result_data["title"][:40]
        description = result_data["description"][:850]

        # 3. 画像をGASに送るためのデータ(Base64)に変換
        encoded_images = []
        for img in target_images:
            img_buf = io.BytesIO()
            img.save(img_buf, format="JPEG", quality=95)
            b64_str = base64.b64encode(img_buf.getvalue()).decode("utf-8")
            encoded_images.append(b64_str)

        safe_folder_name = re.sub(r'[\\/:*?"<>|]', '_', item_text)[:50] or "名称未設定商品"

        # 4. GASのWebアプリURLへデータを送信してドライブに保存
        gas_url = os.environ["GAS_WEBAPP_URL"]
        gas_payload = {
            "folder_name": safe_folder_name,
            "title": title,
            "description": description,
            "images": encoded_images
        }

        gas_res = requests.post(gas_url, json=gas_payload, allow_redirects=True, timeout=60)
        gas_result = gas_res.json()

        if gas_result.get("status") != "success":
            raise HTTPException(status_code=500, detail=f"GAS Error: {gas_result.get('message')}")

        # 5. iPhoneショートカットへ完了結果を返す
        return {
            "status": "success",
            "folder_name": safe_folder_name,
            "title": title,
            "description": description,
            "drive_url": gas_result.get("folderUrl", "")
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
