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
      "status": "Brand Image Processor API (Mercari Pro 10 Rules) is running"
  }


def auto_enhance_and_white_background(img: Image.Image) -> Image.Image:
  """ガンマ補正 ＋ アンシャープマスクで革のシボ感・凹凸・輪郭をパキッと立体的に際立たせる"""
  img = img.convert("RGB")
  GAMMA = 0.62

  lut = []
  for i in range(256):
    norm = i / 255.0
    g_val = math.pow(norm, GAMMA) * 255.0
    if g_val > 165:
      g_val = 255.0
    elif g_val > 125:
      ratio = (g_val - 125) / (165 - 125)
      g_val = 125 + ratio * (255 - 125)
    lut.append(min(255, max(0, int(g_val))))

  enhanced = img.point(lut * 3)
  enhancer_c = ImageEnhance.Contrast(enhanced)
  enhanced = enhancer_c.enhance(1.15)
  enhanced = enhanced.filter(
      ImageFilter.UnsharpMask(radius=2.0, percent=160, threshold=3)
  )
  return enhanced


def crop_center(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
  """中央の被写体を残して綺麗にリサイズ・トリミング"""
  return ImageOps.fit(
      img,
      (target_w, target_h),
      method=Image.Resampling.LANCZOS,
      centering=(0.5, 0.5),
  )


# --- 専門レイアウト結合関数群 ---


def make_1x4_side_strip(images: list[Image.Image]) -> Image.Image:
  """⑥ 側面・マチ用：1行4列（縦分割）"""
  n = len(images)
  cell_w, cell_h = 350, 1100
  collage = Image.new("RGB", (cell_w * n, cell_h), (255, 255, 255))
  for idx, img in enumerate(images):
    fitted = crop_center(img, cell_w, cell_h)
    collage.paste(fitted, (idx * cell_w, 0))
  return collage


def make_2x4_corners_grid(images: list[Image.Image]) -> Image.Image:
  """⑧ 角8箇所用：2列×4行の左右対称グリッド"""
  n = len(images)
  cols = 2
  rows = (n + cols - 1) // cols
  cell_w, cell_h = 600, 300
  collage = Image.new("RGB", (cols * cell_w, rows * cell_h), (255, 255, 255))
  for idx, img in enumerate(images):
    fitted = crop_center(img, cell_w, cell_h)
    r, c = idx // cols, idx % cols
    collage.paste(fitted, (c * cell_w, r * cell_h))
  return collage


def make_stacked_vertical(images: list[Image.Image]) -> Image.Image:
  """⑦ 内側側面見開き / ⑤ ポケット用：上下2〜3段分割"""
  n = len(images)
  cell_w, cell_h = 800, 400
  collage = Image.new("RGB", (cell_w, cell_h * n), (255, 255, 255))
  for idx, img in enumerate(images):
    fitted = crop_center(img, cell_w, cell_h)
    collage.paste(fitted, (0, idx * cell_h))
  return collage


def make_1_big_2_small(images: list[Image.Image]) -> Image.Image:
  """④ 小銭入れ・札入れ用：上部1大＋下部2小のレイアウト"""
  if len(images) < 3:
    return make_stacked_vertical(images)

  canvas_w, canvas_h = 800, 800
  collage = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))

  # 上段大画像（全幅、高さ500）
  top_img = crop_center(images[0], 800, 500)
  collage.paste(top_img, (0, 0))

  # 下段2小画像（各幅400、高さ300）
  b1 = crop_center(images[1], 400, 300)
  b2 = crop_center(images[2], 400, 300)
  collage.paste(b1, (0, 500))
  collage.paste(b2, (400, 500))

  return collage


def make_generic_grid(images: list[Image.Image]) -> Image.Image:
  """⑨ 金具・ロゴ・シリアル等：汎用2x2 / 2x3 グリッド"""
  n = len(images)
  if n == 1:
    return crop_center(images[0], 1000, 1000)
  cols = 2
  rows = (n + cols - 1) // cols
  cell_w, cell_h = 500, 500
  collage = Image.new("RGB", (cols * cell_w, rows * cell_h), (255, 255, 255))
  for idx, img in enumerate(images):
    fitted = crop_center(img, cell_w, cell_h)
    r, c = idx // cols, idx % cols
    collage.paste(fitted, (c * cell_w, r * cell_h))
  return collage


# --- Gemini AI 自動判定関数 ---


def classify_images_with_gemini(images: list[Image.Image]) -> dict:
  """Gemini APIで各画像をメルカリ10ルールに沿って高精度に自動分類"""
  if not client:
    return {"default": list(range(len(images)))}

  prompt = """
あなたはフリマアプリ（メルカリ等）のブランド品出品プロ画像判定AIです。
提供された画像群（インデックス0から順）を解析し、以下の役割カテゴリに厳密に分類してJSONで出力してください。

【カテゴリ一覧】
1. "surface": 表面（正面メイン顔写真、1枚）
2. "back": 裏面（背面写真、1枚）
3. "inside_full": 内側見開き全景（全体が開いた状態、1枚）
4. "coin_bill": 札入れ・小銭入れの内部写真群（2〜4枚）
5. "pockets": カードポケット・内側ポケット内部写真群（2〜3枚）
6. "sides": 側面・マチ・ファスナー4方向写真群（4枚）
7. "inside_sides": 内側側面の折りたたみ・マチの広がり写真群（2枚）
8. "corners": 表裏の角スレ確認用写真群（角4〜8箇所）
9. "hardware_logo": ファスナー引き手・金属パーツ・ロゴ刻印・シリアル/NFC（2〜8枚）
10. "accessories": 箱・保存袋・リボン等の付属品写真（1枚）

【出力形式】
JSONフォーマットのみ（他の説明文は一切出力しないでください）:
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
    contents.append(f"Image index {idx}:")
    thumb = img.copy()
    thumb.thumbnail((400, 400))
    contents.append(thumb)

  try:
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=contents,
        config=types.GenerateContentConfig(
            response_mime_type="application/json", temperature=0.1
        ),
    )
    return json.loads(response.text)
  except Exception as e:
    print(f"Gemini AI Classification Error: {e}")
    return {"default": list(range(len(images)))}


@app.post("/process-images")
async def process_images(request: Request):
  body = await request.body()
  if not body:
    raise HTTPException(status_code=400, detail="データが届いていません")

  images = []
  try:
    with zipfile.ZipFile(io.BytesIO(body)) as z:
      for filename in z.namelist():
        if not filename.startswith("__MACOSX") and not filename.startswith("."):
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
    raise HTTPException(status_code=400, detail="画像データを読み込めませんでした")

  # 1. 画像の高品質補正（ガンマ補正＋白飛ばし＋アンシャープマスク）
  enhanced_images = [auto_enhance_and_white_background(img) for img in images]

  # 2. Gemini AI による画像分類
  cls = classify_images_with_gemini(enhanced_images)

  # 側面(sides)や付属品(accessories)が無い場合の「枠あまり検知」
  has_sides = len(cls.get("sides", [])) > 0
  has_accessories = len(cls.get("accessories", [])) > 0
  flexible_mode = (not has_sides) or (not has_accessories)

  output_files = {}

  # ① 表面（単体 1:1）
  if cls.get("surface"):
    idx = cls["surface"][0]
    output_files["01_surface.jpg"] = crop_center(
        enhanced_images[idx], 1000, 1000
    )

  # ② 裏面（単体 1:1）
  if cls.get("back"):
    idx = cls["back"][0]
    output_files["02_back.jpg"] = crop_center(enhanced_images[idx], 1000, 1000)

  # ③ 内側見開き（単体 1:1）
  if cls.get("inside_full"):
    idx = cls["inside_full"][0]
    output_files["03_inside_full.jpg"] = crop_center(
        enhanced_images[idx], 1000, 1000
    )

  # ④ 札入れ・小銭入れ
  coin_indices = cls.get("coin_bill", [])
  if coin_indices:
    imgs = [
        enhanced_images[i] for i in coin_indices if i < len(enhanced_images)
    ]
    if flexible_mode and len(imgs) > 1:
      # 側面や付属品が無い場合はバラして単体写真として枠を埋める
      for sub_idx, sub_img in enumerate(imgs):
        output_files[f"04_coin_bill_{sub_idx+1}.jpg"] = crop_center(
            sub_img, 1000, 1000
        )
    else:
      # 1大2小などのコラージュ結合
      output_files["04_coin_bill_combined.jpg"] = make_1_big_2_small(imgs)

  # ⑤ 内側ポケット
  pocket_indices = cls.get("pockets", [])
  if pocket_indices:
    imgs = [
        enhanced_images[i] for i in pocket_indices if i < len(enhanced_images)
    ]
    if flexible_mode and len(imgs) > 1:
      for sub_idx, sub_img in enumerate(imgs):
        output_files[f"05_pocket_{sub_idx+1}.jpg"] = crop_center(
            sub_img, 1000, 1000
        )
    else:
      output_files["05_pockets_combined.jpg"] = make_stacked_vertical(imgs)

  # ⑥ 側面4方向（1x4縦分割）
  side_indices = cls.get("sides", [])
  if side_indices:
    imgs = [
        enhanced_images[i] for i in side_indices if i < len(enhanced_images)
    ]
    output_files["06_sides_1x4.jpg"] = make_1x4_side_strip(imgs)

  # ⑦ 内側側面見開き（上下2段分割）
  in_side_indices = cls.get("inside_sides", [])
  if in_side_indices:
    imgs = [
        enhanced_images[i] for i in in_side_indices if i < len(enhanced_images)
    ]
    output_files["07_inside_sides_stacked.jpg"] = make_stacked_vertical(imgs)

  # ⑧ 表裏角8箇所（2x4左右対称）
  corner_indices = cls.get("corners", [])
  if corner_indices:
    imgs = [
        enhanced_images[i] for i in corner_indices if i < len(enhanced_images)
    ]
    output_files["08_corners_2x4.jpg"] = make_2x4_corners_grid(imgs)

  # ⑨ 金具・ロゴ・刻印・シリアル/NFC
  hw_indices = cls.get("hardware_logo", [])
  if hw_indices:
    imgs = [enhanced_images[i] for i in hw_indices if i < len(enhanced_images)]
    output_files["09_hardware_logo_combined.jpg"] = make_generic_grid(imgs)

  # ⑩ 付属品（単体 1:1）
  acc_indices = cls.get("accessories", [])
  if acc_indices:
    idx = acc_indices[0]
    output_files["10_accessories.jpg"] = crop_center(
        enhanced_images[idx], 1000, 1000
    )

  # フォールバック（分類不能時）
  if not output_files:
    output_files["01_processed_collage.jpg"] = make_generic_grid(
        enhanced_images
    )

  # 完成画像をZIP圧縮してレスポンス返却
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
