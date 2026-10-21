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
          "Brand Image Processor API (AI & Natural Body Pure White BG Model) is"
          " running"
      )
  }


def auto_enhance_and_white_background(img: Image.Image) -> Image.Image:
  """商品本体の自然な色味・質感は100%保持し、背景のグレーだけを爽やかな純白にする補正"""
  img = img.convert("RGB")

  # 1. 本体（中間色）の色味を完璧に保ちつつ、背景領域（ハイライト）のみ純白化するカーブ
  lut = []
  for i in range(256):
    x = i / 255.0

    if x < 0.35:
      # 暗部（影）の救済
      y = math.pow(x, 0.50) * 1.10
    elif x < 0.72:
      # 商品本体（キャメル・ブラウン）の中間色は現在の最高のバランスを完璧にキープ
      t = (x - 0.35) / (0.72 - 0.35)
      y_start = math.pow(0.35, 0.50) * 1.10
      y = y_start + t * (0.86 - y_start)
    else:
      # 背景の薄グレー領域をスムーズに純白（1.0）へ収束させる
      t = (x - 0.72) / (1.0 - 0.72)
      y = 0.86 + math.sin(t * math.pi / 2) * (1.0 - 0.86)

    val = min(255, max(0, int(y * 255.0)))

    # 背景領域を抜いてパッと明るい純白「255」にする（本体の色飛びを防ぐ絶妙な閾値）
    if val > 218:
      val = 255

    lut.append(val)

  enhanced = img.point(lut * 3)

  # 2. 自然で好評価だった彩度・コントラストのバランスをそのまま維持
  enhancer_s = ImageEnhance.Color(enhanced)
  enhanced = enhancer_s.enhance(1.30)

  enhancer_c = ImageEnhance.Contrast(enhanced)
  enhanced = enhancer_c.enhance(1.12)

  # 3. 革のシボ感・ステッチ・角の立体感を強調
  return enhanced.filter(
      ImageFilter.UnsharpMask(radius=1.2, percent=135, threshold=2)
  )


def crop_center(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
  return ImageOps.fit(
      img,
      (target_w, target_h),
      method=Image.Resampling.LANCZOS,
      centering=(0.5, 0.5),
  )


# --- コラージュ生成 ---


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


def classify_images_with_gemini(images: list[Image.Image]) -> dict:
  """超軽量・明確なインデックス指定による Gemini AI 役割判別"""
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
以下のキーを持ったJSONフォーマットのみ:
{
  "surface": [0],
  "back": [1],
  "inside_full": [2],
  "coin_bill": [3, 4, 5],
  "pockets": [6, 7],
  "sides": [8, 9, 10, 11],
  "inside_sides": [12, 13],
  "corners": [14, 15, 16, 17, 18, 19, 20, 21],
  "hardware_logo": [22, 23, 24],
  "accessories": [25]
}
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

    # 1. 画像補正（本体質感保持＋背景純白化）
    enhanced_images = [auto_enhance_and_white_background(img) for img in images]

    # 2. AI分類
    cls = classify_images_with_gemini(enhanced_images)

    output_files = {}

    def get_imgs(key):
      indices = cls.get(key, [])
      return [enhanced_images[i] for i in indices if i < len(enhanced_images)]

    # 3. 分類に基づいた個別 / 結合画像の生成
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

    has_sides = len(get_imgs("sides")) > 0
    has_accessories = len(get_imgs("accessories")) > 0
    flexible_mode = (not has_sides) or (not has_accessories)

    # ④ 札入れ・小銭入れ
    coin_imgs = get_imgs("coin_bill")
    if coin_imgs:
      if flexible_mode and len(coin_imgs) > 1:
        for s_idx, s_img in enumerate(coin_imgs):
          output_files[f"04_coin_bill_{s_idx+1}.jpg"] = crop_center(
              s_img, 1000, 1000
          )
      else:
        output_files["04_coin_bill_combined.jpg"] = make_1_big_2_small(
            coin_imgs
        )

    # ⑤ 内側ポケット
    pocket_imgs = get_imgs("pockets")
    if pocket_imgs:
      if flexible_mode and len(pocket_imgs) > 1:
        for s_idx, s_img in enumerate(pocket_imgs):
          output_files[f"05_pockets_{s_idx+1}.jpg"] = crop_center(
              s_img, 1000, 1000
          )
      else:
        output_files["05_pockets_combined.jpg"] = make_stacked_vertical(
            pocket_imgs
        )

    # ⑥ 側面
    if get_imgs("sides"):
      output_files["06_sides_1x4.jpg"] = make_1x4_side_strip(get_imgs("sides"))

    # ⑦ 内側側面見開き
    if get_imgs("inside_sides"):
      output_files["07_inside_sides_stacked.jpg"] = make_stacked_vertical(
          get_imgs("inside_sides")
      )

    # ⑧ 表裏角8箇所
    if get_imgs("corners"):
      output_files["08_corners_2x4.jpg"] = make_2x4_corners_grid(
          get_imgs("corners")
      )

    # ⑨ 金具・ロゴ
    if get_imgs("hardware_logo"):
      output_files["09_hardware_logo_combined.jpg"] = make_generic_grid(
          get_imgs("hardware_logo")
      )

    # ⑩ 付属品
    if get_imgs("accessories"):
      output_files["10_accessories.jpg"] = crop_center(
          get_imgs("accessories")[0], 1000, 1000
      )

    # 【AI仕分け失敗時のフォールバック】
    if not output_files:
      for idx, img in enumerate(enhanced_images):
        output_files[f"{idx+1:02d}_processed_photo.jpg"] = crop_center(
            img, 1000, 1000
        )

    # 4. ZIP圧縮
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
      for fname, img_obj in output_files.items():
        img_buf = io.BytesIO()
        img_obj.save(img_buf, format="JPEG", quality=95)
        zf.writestr(fname, img_buf.getvalue())

    zip_buf.seek(0)
    return Response(
        content=zip_buf.getvalue(),
        media_type="attachment; filename=processed_images.zip",
        headers={
            "Content-Disposition": "attachment; filename=processed_images.zip"
        },
    )

  except Exception as e:
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
      zf.writestr(
          "00_error_notice.txt",
          f"処理でエラーが起きたため、バラで補正済み画像を返却します: {str(e)}",
      )
      for idx, img in enumerate(enhanced_images):
        img_buf = io.BytesIO()
        img.save(img_buf, format="JPEG", quality=90)
        zf.writestr(f"{idx+1:02d}_backup_photo.jpg", img_buf.getvalue())

    zip_buf.seek(0)
    return Response(
        content=zip_buf.getvalue(),
        media_type="attachment; filename=processed_images.zip",
    )
