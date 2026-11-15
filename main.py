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
    description: str = Field(
        description="メルカリ用商品説明文（指定フォーマット通り。必ず950文字以内。生産国は記載しない）"
    )


@app.get("/")
def read_root():
    return {"status": "Brand Image & Listing Processor is running"}


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
        if len(image_parts) < 10:
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
4. 画像から確認できる範囲で丁寧かつ誠実に記載し、確認できないシリアル番号・サイズ・付属品は勝手に捏造しないこと。
5. 「生産国（原産国）」の項目は記載しないこと。
6. シリアルナンバーが画像から読み取れる場合は「・シリアルナンバー：〇〇（写真〇枚目参照）」と記載し、画像に写っていない・読み取れない場合は「・シリアルナンバー：」の行ごと削除すること。
7. 付属品が画像に写っている場合は「・付属品　：箱、保存袋(写真〇枚目)」のように何枚目に写っているかも記載し、本体のみの場合は「・付属品　：本体のみ」とすること。
8. 最後の行は「管理番号111」のようにコロン（：）を付けずに入力情報の管理番号を記載すること。

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
・サイズ　：横幅約〇〇cm × 縦約〇〇cm × マチ約〇〇cm（※画像や情報から推定できる場合のみ記載、不明なら行ごと削除）
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

    return {
        "status": "success",
        "folder_name": folder_name,
        "title": title,
        "folderUrl": gas_json.get("folderUrl", ""),
    }
