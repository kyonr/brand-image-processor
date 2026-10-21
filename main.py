import io
import json
import math
import os
import zipfile
from fastapi import FastAPI, HTTPException, Request, Response
from google import genai
from google.genai import types
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageStat

app = FastAPI()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None


@app.get("/")
def read_root():
  return {
      "status": (
          "Brand Image Processor API (Auto Black Detection & Brilliance"
          " Model) is running"
      )
  }


# === 黒系商品の自動判定関数 ===
def is_black_item(img: Image.Image) -> bool:
  """商品本体が黒・ダーク系かどうかを自動判別"""
  # 画像中央エリア（商品が存在する部分）を抽出
  w, h = img.size
  crop_area = img.crop((int(w * 0.2), int(h * 0.2), int(w * 0.8), int(h * 0.8)))

  # 背景の白（輝度200以上）を除外した「商品領域」の平均輝度を測定
  gray = crop_area.convert("L")
  pixels = [p for p in list(gray.getdata()) if p < 210]

  if not pixels:
    return False

  avg_darkness = sum(pixels) / len(pixels)
  # 平均輝度が低い（暗い）場合は黒系アイテムと判断
  return avg_darkness < 85


# === 1. 明るさ＆色彩補正（黒自動検出機能付き） ===
def auto_enhance_and_white_background(img: Image.Image) -> Image.Image:
  img = img.convert("RGB")
  is_black = is_black_item(img)

  lut = []
  for i in range(256):
    x = i / 255.0

    if is_black:
      # 【黒系商品専用のトーンカーブ】
      # 暗部はしっかり見せつつ、テカリ・白光りを防ぐためにハイライト領域を少し抑える（ブリリアンスマイナス効果）
      if x < 0.60:
        y = math.pow(x, 0.55) * 1.08
      elif x < 0.88:
        t = (x - 0.60) / (0.88 - 0.60)
        y_60 = math.pow(0.60, 0.55) * 1.08
        y = y_60 + t * (0.94 - y_60)
      else:
        t = (x - 0.88) / 0.12
        y = 0.94 + t * 0.06
    else:
      # 【通常カラー用（パステル・ブラウン・ベージュ用）従来通りの最高の計算式】
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

  if is_black:
    # 黒アイテム：コントラストを強め（1.18倍）にして黒を引き締め、白光りを抑える
    enhancer_c = ImageEnhance.Contrast(enhanced)
    enhanced = enhancer_c.enhance(1.18)

    enhancer_s = ImageEnhance.Color(enhanced)
    enhanced = enhancer_s.enhance(1.15)
  else:
    # 通常カラー：従来の完璧な設定
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


# === 2. スマート・コラージュレイアウトエンジン ===
def make_1x4_side_strip(images: list[Image.Image]) -> Image.Image:
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


def make_smart_grid(
    images: list[Image.Image], target_size: int = 1000
) -> Image.Image:
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
  elif n <= 9:
    cols, rows = 3, 3
  else:
    cols, rows = 4, (n + 3) // 4

  cell_w = target_size // cols
  cell_h = target_size // rows

  collage = Image.new("RGB", (target_size, target_size), (255, 255, 255))
  for idx, img in enumerate(images):
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

    # 1. 自動判別付きの補正を全画像に適用
    enhanced_images = [auto_enhance_and_white_background(img) for img in images]

    # 2. AI分類
    cls = classify_images_with_gemini(enhanced_images)

    output_files = {}

    def get_imgs_guarded(key, want_portrait=False):
      indices = cls.get(key, [])
      res = []
      for i in indices:
        if i < len(enhanced_images):
          img = enhanced_images[i]
          w, h = img.size
          is_portrait = h > w * 1.12
          if want_portrait and is_portrait:
            res.append(img)
          elif not want_portrait and not is_portrait:
            res.append(img)
      return res

    # メイン写真
    if get_imgs_guarded("surface"):
      output_files["01_surface.jpg"] = crop_center(
          get_imgs_guarded("surface")[0], 1000, 1000
      )
    if get_imgs_guarded("back"):
      output_files["02_back.jpg"] = crop_center(
          get_imgs_guarded("back")[0], 1000, 1000
      )
    if get_imgs_guarded("inside_full"):
      output_files["03_inside_full.jpg"] = crop_center(
          get_imgs_guarded("inside_full")[0], 1000, 1000
      )

    # ⑥ 側面
    side_imgs = get_imgs_guarded("sides", want_portrait=True)
    if side_imgs:
      output_files["06_sides_1x4.jpg"] = make_1x4_side_strip(side_imgs)

    # ⑧ 角スレ＆詳細パーツ統合
    corner_imgs = get_imgs_guarded("corners", want_portrait=False)
    hardware_imgs = get_imgs_guarded("hardware_logo", want_portrait=False)
    all_square_detail_imgs = corner_imgs + hardware_imgs

    if all_square_detail_imgs:
      if len(all_square_detail_imgs) <= 8:
        output_files["08_corners_combined.jpg"] = make_smart_grid(
            all_square_detail_imgs
        )
      else:
        if corner_imgs:
          output_files["08_corners_combined.jpg"] = make_smart_grid(
              corner_imgs
          )
        if hardware_imgs:
          output_files["09_hardware_logo_combined.jpg"] = make_smart_grid(
              hardware_imgs
          )

    # その他パーツ
    if get_imgs_guarded("coin_bill"):
      output_files["04_coin_bill_combined.jpg"] = make_smart_grid(
          get_imgs_guarded("coin_bill")
      )
    if get_imgs_guarded("pockets"):
      output_files["05_pockets_combined.jpg"] = make_stacked_vertical(
          get_imgs_guarded("pockets")
      )
    if get_imgs_guarded("accessories"):
      output_files["10_accessories.jpg"] = crop_center(
          get_imgs_guarded("accessories")[0], 1000, 1000
      )

    # スマートフォールバック
    if not output_files:
      sides_list = []
      corners_list = []

      for img in enhanced_images:
        w, h = img.size
        if h > w * 1.12:
          sides_list.append(img)
        else:
          corners_list.append(img)

      if sides_list:
        output_files["06_sides_combined.jpg"] = make_1x4_side_strip(
            sides_list
        )
      if corners_list:
        output_files["08_corners_combined.jpg"] = make_smart_grid(
            corners_list
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
