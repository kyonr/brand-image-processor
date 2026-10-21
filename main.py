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
          "Brand Image Processor API (Perfect Black Tone & Dedicated Grid"
          " Collage) is running"
      )
  }


# === 黒・暗色系商品の自動判定関数 ===
def is_dark_item(img: Image.Image) -> bool:
  """商品本体が黒・紺・ダークブラウン等か自動判定"""
  w, h = img.size
  crop_area = img.crop((int(w * 0.25), int(h * 0.25), int(w * 0.75), int(h * 0.75)))
  gray = crop_area.convert("L")

  # 背景の白（輝度210以上）を除外した「商品領域」の画素を取り出す
  pixels = [p for p in list(gray.getdata()) if p < 210]

  if not pixels:
    return False

  avg_darkness = sum(pixels) / len(pixels)
  return avg_darkness < 75


# === 1. 明るさ＆色彩補正（全カラー万能 ＋ 黒・暗色系専用トーンカーブ） ===
def auto_enhance_and_white_background(img: Image.Image) -> Image.Image:
  img = img.convert("RGB")
  dark_mode = is_dark_item(img)

  lut = []
  for i in range(256):
    x = i / 255.0

    if dark_mode:
      # 【黒・紺・ダークカラー専用トーンカーブ】
      # 黒色を深く保ち、白光りを防ぎつつ背景だけをすっきり白くする
      if x < 0.30:
        y = math.pow(x, 1.05) * 0.95
      elif x < 0.70:
        t = (x - 0.30) / (0.70 - 0.30)
        y_30 = math.pow(0.30, 1.05) * 0.95
        y = y_30 + t * (0.82 - y_30)
      elif x < 0.92:
        t = (x - 0.70) / (0.92 - 0.70)
        y = 0.82 + math.sin(t * math.pi / 2) * (0.98 - 0.82)
      else:
        t = (x - 0.92) / 0.08
        y = 0.98 + t * 0.02
    else:
      # 【通常カラー（パステル・ブラウン・ベージュ等）高評価獲得済みの設定】
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
    # 黒系：コントラストを高めて（1.22倍）深い黒とシャープな質感を両立
    enhancer_c = ImageEnhance.Contrast(enhanced)
    enhanced = enhancer_c.enhance(1.22)
    enhancer_s = ImageEnhance.Color(enhanced)
    enhanced = enhancer_s.enhance(1.05)
  else:
    # 通常カラー：高評価の色彩＆コントラスト
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


# === 2. 理想のレイアウト生成エンジン ===


def make_1x4_side_strip(images: list[Image.Image]) -> Image.Image:
  """側面用：1000x1000正方形キャンバスに縦長写真を横一列に並べる（image_59.png と完全一致）"""
  if not images:
    return None
  n = len(images)
  if n == 1:
    return crop_center(images[0], 1000, 1000)

  cell_w = 1000 // n
  cell_h = 1000
  collage = Image.new("RGB", (1000, 1000), (255, 255, 255))
  for idx, img in enumerate(images):
    collage.paste(crop_center(img, cell_w, cell_h), (idx * cell_w, 0))
  return collage


def make_corners_2x4_grid(
    images: list[Image.Image], target_size: int = 1000
) -> Image.Image:
  """角スレ・詳細パーツ用：4列x2行 (2x4) レイアウト（image_60.png と完全一致）"""
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
  else:
    # 7〜8枚（およびそれ以上）：絶対 4列×2行（上4枚・下4枚）
    cols, rows = 4, 2

  cell_w = target_size // cols
  cell_h = target_size // rows

  collage = Image.new("RGB", (target_size, target_size), (255, 255, 255))
  for idx, img in enumerate(images[: cols * rows]):
    r, c = idx // cols, idx % cols
    collage.paste(crop_center(img, cell_w, cell_h), (c * cell_w, r * cell_h))
  return collage


def make_stacked_vertical(images: list[Image.Image]) -> Image.Image:
  if not images:
    return None
  n = len(images)
  if n == 1:
    return crop_center(images[0], 1000, 1000)

  cell_w = 1000
  cell_h = 1000 // n
  collage = Image.new("RGB", (1000, 1000), (255, 255, 255))
  for idx, img in enumerate(images):
    collage.paste(crop_center(img, cell_w, cell_h), (0, idx * cell_h))
  return collage


# === 3. AI（Gemini）判別機能 ===
def classify_images_with_gemini(images: list[Image.Image]) -> dict:
  if not client:
    return {}

  prompt = """
あなたはフリマアプリ（メルカリ等）のブランド品出品プロ画像判定AIです。
提供された画像群を解析し、以下の役割カテゴリに厳密に分類してJSONで出力してください。

【カテゴリ一覧】
- "surface": 表面（1枚）
- "back": 裏面（1枚）
- "inside_full": 内側見開き全景（1枚）
- "coin_bill": 札入れ・小銭入れの内部
- "pockets": カードポケット内部
- "sides": 側面・マチ（縦長ショット4枚など）
- "corners": 角スレ・フチ確認用（角4〜8箇所）
- "hardware_logo": 金属パーツ・ロゴ刻印・シリアル
- "accessories": 付属品

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

    # 1. 自動カラー補正（黒・紺引き締めモード含む）
    enhanced_images = [auto_enhance_and_white_background(img) for img in images]

    # 2. 画像のアスペクト比（縦横比）で確実に側面と角を分離！
    portrait_sides = []  # 縦長写真 ➔ 1x4側面結合用
    square_corners = []  # 四角・横長写真 ➔ 2x4角スレ結合用

    for img in enhanced_images:
      w, h = img.size
      if h > w * 1.08:  # 縦長ショット
        portrait_sides.append(img)
      else:  # 四角・角ショット
        square_corners.append(img)

    output_files = {}

    # 3. AI判定も組み合わせて結合を生成
    cls = classify_images_with_gemini(enhanced_images)

    def get_imgs_by_key(key):
      indices = cls.get(key, [])
      return [enhanced_images[i] for i in indices if i < len(enhanced_images)]

    # メイン単体写真
    if get_imgs_by_key("surface"):
      output_files["01_surface.jpg"] = crop_center(
          get_imgs_by_key("surface")[0], 1000, 1000
      )
    if get_imgs_by_key("back"):
      output_files["02_back.jpg"] = crop_center(
          get_imgs_by_key("back")[0], 1000, 1000
      )
    if get_imgs_by_key("inside_full"):
      output_files["03_inside_full.jpg"] = crop_center(
          get_imgs_by_key("inside_full")[0], 1000, 1000
      )

    # 側面（縦長ショット） ➔ 必ず 1x4 の横一列ストリップ画像（image_59.png）に結合！
    if portrait_sides:
      output_files["06_sides_1x4.jpg"] = make_1x4_side_strip(portrait_sides)

    # 角・詳細パーツ（四角ショット） ➔ 必ず 2x4 (4列2行) グリッド画像（image_60.png）に結合！
    if square_corners:
      main_used = (
          get_imgs_by_key("surface")
          + get_imgs_by_key("back")
          + get_imgs_by_key("inside_full")
      )
      corners_to_combine = [
          img for img in square_corners if img not in main_used
      ]

      if not corners_to_combine:
        corners_to_combine = square_corners

      output_files["08_corners_2x4.jpg"] = make_corners_2x4_grid(
          corners_to_combine
      )

    # その他内部パーツ
    if get_imgs_by_key("coin_bill"):
      output_files["04_coin_bill_combined.jpg"] = make_corners_2x4_grid(
          get_imgs_by_key("coin_bill")
      )
    if get_imgs_by_key("pockets"):
      output_files["05_pockets_combined.jpg"] = make_stacked_vertical(
          get_imgs_by_key("pockets")
      )
    if get_imgs_by_key("accessories"):
      output_files["10_accessories.jpg"] = crop_center(
          get_imgs_by_key("accessories")[0], 1000, 1000
      )

    # 4. ZIP圧縮＆返却
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
      for fname, img_obj in output_files.items():
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
      zf.writestr(
          "00_error_notice.txt",
          f"処理エラー回避のため個別保存しました: {str(e)}",
      )
      for idx, img in enumerate(enhanced_images):
        img_buf = io.BytesIO()
        img.save(img_buf, format="JPEG", quality=90)
        zf.writestr(f"{idx+1:02d}_backup_photo.jpg", img_buf.getvalue())

    zip_buf.seek(0)
    return Response(content=zip_buf.getvalue(), media_type="application/zip")
