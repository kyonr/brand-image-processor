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
          "Brand Image Processor API (Perfect Brightness + Smart Auto Collage"
          " Model) is running"
      )
  }


# === 1. 完璧に仕上がった明るさ＆色彩補正（全カラー万能モデル） ===
def auto_enhance_and_white_background(img: Image.Image) -> Image.Image:
  """全カラー対応（背景明瞭化 ＋ ブラウン暗さ解消 ＋ パステル白飛び防止）"""
  img = img.convert("RGB")

  lut = []
  for i in range(256):
    x = i / 255.0

    # ゾーン1: 暗部〜中間色 (x < 0.68)
    if x < 0.68:
      y = math.pow(x, 0.52) * 1.12
    # ゾーン2: 背景の薄グレー領域 (0.68 <= x < 0.92)
    elif x < 0.92:
      t = (x - 0.68) / (0.92 - 0.68)
      y_68 = math.pow(0.68, 0.52) * 1.12
      y = y_68 + math.sin(t * math.pi / 2) * (0.98 - y_68)
    # ゾーン3: 最高輝度ハイライト部 (x >= 0.92)
    else:
      t = (x - 0.92) / 0.08
      y = 0.98 + t * 0.02

    val = min(255, max(0, int(y * 255.0)))
    lut.append(val)

  enhanced = img.point(lut * 3)

  # 色彩＆コントラスト最適化
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


# === 2. コラージュレイアウト生成エンジン ===
def make_1x4_side_strip(images: list[Image.Image]) -> Image.Image:
  n = len(images)
  cell_w, cell_h = 350, 1100
  collage = Image.new("RGB", (cell_w * n, cell_h), (255, 255, 255))
  for idx, img in enumerate(images):
    collage.paste(crop_center(img, cell_w, cell_h), (idx * cell_w, 0))
  return collage


def make_2x4_corners_grid(images: list[Image.Image]) -> Image.Image:
  n = len(images)
  cols = 2
  rows = (n + cols - 1) // cols
  cell_w, cell_h = 600, 300
  collage = Image.new("RGB", (cols * cell_w, rows * cell_h), (255, 255, 255))
  for idx, img in enumerate(images):
    r, c = idx // cols, idx % cols
    collage.paste(crop_center(img, cell_w, cell_h), (c * cell_w, r * cell_h))
  return collage


def make_stacked_vertical(images: list[Image.Image]) -> Image.Image:
  n = len(images)
  cell_w, cell_h = 800, 400
  collage = Image.new("RGB", (cell_w, cell_h * n), (255, 255, 255))
  for idx, img in enumerate(images):
    collage.paste(crop_center(img, cell_w, cell_h), (0, idx * cell_h))
  return collage


def make_1_big_2_small(images: list[Image.Image]) -> Image.Image:
  if len(images) < 3:
    return make_stacked_vertical(images)
  collage = Image.new("RGB", (800, 800), (255, 255, 255))
  collage.paste(crop_center(images[0], 800, 500), (0, 0))
  collage.paste(crop_center(images[1], 400, 300), (0, 500))
  collage.paste(crop_center(images[2], 400, 300), (400, 500))
  return collage


def make_generic_grid(images: list[Image.Image]) -> Image.Image:
  n = len(images)
  if n == 1:
    return crop_center(images[0], 1000, 1000)
  cols = 2
  rows = (n + cols - 1) // cols
  cell_w, cell_h = 500, 500
  collage = Image.new("RGB", (cols * cell_w, rows * cell_h), (255, 255, 255))
  for idx, img in enumerate(images):
    r, c = idx // cols, idx % cols
    collage.paste(crop_center(img, cell_w, cell_h), (c * cell_w, r * cell_h))
  return collage


# === 3. AI（Gemini）分類機能 ===
def classify_images_with_gemini(images: list[Image.Image]) -> dict:
  if not client:
    return {}

  prompt = """
あなたはフリマアプリ（メルカリ等）のブランド品出品プロ画像判定AIです。
提供された画像群を解析し、以下の役割カテゴリに厳密に分類してJSONで出力してください。

【カテゴリ一覧】
- "surface": 表面（正面メイン顔写真、1枚）
- "back": 裏面（背面写真、1枚）
- "inside_full": 内側見開き全景（全体が開いた状態、1枚）
- "coin_bill": 札入れ・小銭入れの内部写真群（2〜4枚）
- "pockets": カードポケット・内側ポケット内部写真群（2〜3枚）
- "sides": 側面・マチ・ファスナー4方向写真群（4枚）
- "inside_sides": 内側側面の折りたたみ・マチの広がり写真群（2枚）
- "corners": 表裏の角スレ確認用写真群（角4〜8箇所）
- "hardware_logo": ファスナー引き手・金属パーツ・ロゴ刻印・シリアル/NFC（2〜8枚）
- "accessories": 箱・保存袋・リボン等の付属品写真（1枚）

【出力形式】
JSONフォーマットのみを返してください。
"""

  contents = [prompt]
  for idx, img in enumerate(images):
    thumb = img.copy()
    thumb.thumbnail((120, 120))
    contents.append(f"[インデックス: {idx}]")
    contents.append(thumb)

  try:
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=contents,
        config=types.GenerateContentConfig(
            response_mime_type="application/json", temperature=0.1
        ),
    )
    text_res = response.text.strip()
    if text_res.startswith("```"):
      text_res = text_res.split("```")[1]
      if text_res.startswith("json"):
        text_res = text_res[4:]
    return json.loads(text_res.strip())
  except Exception as e:
    print(f"Gemini error: {e}")
    return {}


# === 4. メインAPIエンドポイント ===
@app.post("/process-images")
async def process_images(request: Request):
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

    # 1. 完成された最高の明るさ＆色彩補正を全画像に適用
    enhanced_images = [auto_enhance_and_white_background(img) for img in images]

    # 2. AIによる分類判定
    cls = classify_images_with_gemini(enhanced_images)

    output_files = {}

    def get_imgs(key):
      indices = cls.get(key, [])
      return [enhanced_images[i] for i in indices if i < len(enhanced_images)]

    # 3. AI分類に基づく結合画像の生成
    if get_imgs("surface"):
      output_files["01_surface.jpg"] = crop_center(
          get_imgs("surface")[0], 1000, 1000
      )
    if get_imgs("back"):
      output_files["02_back.jpg"] = crop_center(get_imgs("back")[0], 1000, 1000)
    if get_imgs("inside_full"):
      output_files["03_inside_full.jpg"] = crop_center(
          get_imgs("inside_full")[0], 1000, 1000
      )

    if get_imgs("coin_bill"):
      output_files["04_coin_bill_combined.jpg"] = make_1_big_2_small(
          get_imgs("coin_bill")
      )
    if get_imgs("pockets"):
      output_files["05_pockets_combined.jpg"] = make_stacked_vertical(
          get_imgs("pockets")
      )
    if get_imgs("sides"):
      output_files["06_sides_1x4.jpg"] = make_1x4_side_strip(get_imgs("sides"))
    if get_imgs("inside_sides"):
      output_files["07_inside_sides_stacked.jpg"] = make_stacked_vertical(
          get_imgs("inside_sides")
      )
    if get_imgs("corners"):
      output_files["08_corners_2x4.jpg"] = make_2x4_corners_grid(
          get_imgs("corners")
      )
    if get_imgs("hardware_logo"):
      output_files["09_hardware_logo_combined.jpg"] = make_generic_grid(
          get_imgs("hardware_logo")
      )
    if get_imgs("accessories"):
      output_files["10_accessories.jpg"] = crop_center(
          get_imgs("accessories")[0], 1000, 1000
      )

    # 【強靭な安全装置】AI判定が失敗・不足していた場合でも必ず結合画像を作成するフォールバック
    if len(output_files) <= 2 and len(enhanced_images) >= 4:
      output_files = {}  # 初期化してスマート自動振り分け結合を開始

      # 1枚目〜3枚目：メイン（表面・裏面・内側）
      if len(enhanced_images) >= 1:
        output_files["01_surface.jpg"] = crop_center(
            enhanced_images[0], 1000, 1000
        )
      if len(enhanced_images) >= 2:
        output_files["02_back.jpg"] = crop_center(
            enhanced_images[1], 1000, 1000
        )
      if len(enhanced_images) >= 3:
        output_files["03_inside_full.jpg"] = crop_center(
            enhanced_images[2], 1000, 1000
        )

      # 4枚目以降を4枚ずつ／8枚ずつグループ化して必ず1x4・2x4に自動結合！
      remaining = enhanced_images[3:]
      if len(remaining) >= 4:
        output_files["06_sides_1x4.jpg"] = make_1x4_side_strip(remaining[:4])
        remaining = remaining[4:]

      if len(remaining) >= 4:
        output_files["08_corners_2x4.jpg"] = make_2x4_corners_grid(
            remaining[:8]
        )
        remaining = remaining[8:]

      if remaining:
        output_files["09_details_grid.jpg"] = make_generic_grid(remaining)

    # 4. ZIP圧縮＆レスポンス返却
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
      for fname, img_obj in output_files.items():
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
      zf.writestr(
          "00_error_notice.txt",
          f"処理エラー回避のため標準結合を適用しました: {str(e)}",
      )
      for idx, img in enumerate(enhanced_images):
        img_buf = io.BytesIO()
        img.save(img_buf, format="JPEG", quality=90)
        zf.writestr(f"{idx+1:02d}_backup_photo.jpg", img_buf.getvalue())

    zip_buf.seek(0)
    return Response(content=zip_buf.getvalue(), media_type="application/zip")
