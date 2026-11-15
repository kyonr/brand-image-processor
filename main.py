import base64
import io
import os
import re
import zipfile
from fastapi import FastAPI, HTTPException, Request
from google import genai
from google.genai import types
from PIL import Image, ImageOps
from pydantic import BaseModel, Field
import requests

app = FastAPI()


class ListingOutput(BaseModel):
    title: str = Field(description="メルカリ用商品タイトル（必ず40文字以内）")
    description: str = Field(description="メルカリ用商品説明文（必ず850文字以内。素材・生産国の項目は含めない）")


@app.get("/")
def read_root():
    return {"status": "Brand Image & Listing Processor is running"}


def sanitize_folder_name(name: str) -> str:
    cleaned = re.sub(r'[\\/:*?"<>|\r\n]+', " ", name).strip()
    return cleaned[:80] if cleaned else "新規商品"


@app.post("/prepare-listing")
async def prepare_listing(request: Request):
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    gas_webapp_url = os.environ.get("GAS_WEBAPP_URL")

    if not gemini_api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY が設定されていません")
    if not gas_webapp_url:
        raise HTTPException(status_code=500, detail="GAS_WEBAPP_URL が設定されていません")

    body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail="ZIPデータが空です")

    try:
        zip_file = zipfile.ZipFile(io.BytesIO(body))
    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="ZIPファイルの形式が正しくありません")

    info_text = ""
    image_entries = []

    for entry in sorted(zip_file.namelist()):
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
        raise HTTPException(status_code=400, detail="ZIP内に info.txt が見つかりませんでした")
    if not image_entries:
        raise HTTPException(status_code=400, detail="ZIP内に画像ファイルが見つかりませんでした")

    image_parts = []
    base64_images = []

    for entry in image_entries:
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
        if len(image_parts) < 10:
            image_parts.append(
                types.Part.from_bytes(data=jpeg_bytes, mime_type="image/jpeg")
            )

    prompt = f"""あなたは中古ブランド品販売のプロの出品担当者です。
入力情報と商品画像を確認し、メルカリ出品用の「タイトル」と「商品説明文」を作成してください。

【入力情報（管理番号・商品名など）】
{info_text}

【厳守ルール】
1. タイトルは全角40文字以内に必ず収めること（入力情報のキャッチコピーやブランド名・型番・商品名・カラー等を活かし、検索されやすくまとめる）。
2. 商品説明文は全体で850文字以内に必ず収めること。
3. 「素材」「生産国（原産国）」の項目は絶対に記載しないこと。
4. 画像から確認できる範囲で丁寧かつ誠実に記載し、確認できない仕様や付属品は勝手に捏造しないこと。
5. 商品説明文は以下のテンプレート構成に沿って出力すること：

【商品説明文テンプレート】
ご覧いただきありがとうございます！

■ブランド
（ブランド名）

■商品名・型番
（商品名・型番）

■カラー
（カラー）

■サイズ
（画像や情報から分かる範囲のサイズ。不明な場合は省略可）

■仕様
（札入れ・小銭入れ・カードポケット数など画像から分かる仕様）

■付属品
（画像に写っている箱・保存袋など。本体のみの場合は「本体のみ」）

■状態
（画像から読み取れる角スレ・金具・内側・全体コンディションを簡潔に記載）

■購入先
古物商許可証を取得しているブランド専門オークションにて購入した鑑定済みの正規品です。

※中古品にご理解のある方のみご購入をお願いいたします。
※すり替え防止のため、返品は原則お受けしておりません。

管理番号：（入力情報にある管理番号）
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
    description = result.description.strip()[:850]
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

    return {
        "status": "success",
        "folder_name": folder_name,
        "title": title,
        "folderUrl": gas_json.get("folderUrl", ""),
    }
