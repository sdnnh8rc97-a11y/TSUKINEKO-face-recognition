import base64
import io
import numpy as np
from PIL import Image
from fastapi import FastAPI

# --------------------------------------------------
# 匯入你原本的 predict_single / predict_group
# --------------------------------------------------
from src.predictor import predictor


app = FastAPI()


# --------------------------------------------------
# 避免 numpy 物件讓 FastAPI JSON 編碼爆炸
# --------------------------------------------------
def safe_convert(o):
    import numpy as np

    if isinstance(o, (np.int64, np.int32, np.uint8)):
        return int(o)
    if isinstance(o, (np.float32, np.float64)):
        return float(o)
    if isinstance(o, dict):
        return {k: safe_convert(v) for k, v in o.items()}
    if isinstance(o, list):
        return [safe_convert(v) for v in o]
    return o


# --------------------------------------------------
# 測試用根路由
# --------------------------------------------------
@app.get("/")
def root():
    return {"status": "ok", "message": "TSUKINEKO face API running"}


# --------------------------------------------------
# 單張人臉辨識
# --------------------------------------------------
@app.post("/predict")
async def predict(payload: dict):

    # 1. 取出 base64
    img_b64 = payload["image_base64"]

    # 2. base64 → image array
    img_bytes = base64.b64decode(img_b64)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img_np = np.array(img)

    # 3. 執行辨識
    raw_result = predictor.predict_single(img_np)

    # 4. 轉換 numpy → python int/float
    return safe_convert({
        "status": "ok",
        "record_id": payload.get("record_id", "unknown"),
        "result": raw_result
    })


# --------------------------------------------------
# 團體照辨識
# --------------------------------------------------
@app.post("/predict_group")
async def predict_group(payload: dict):

    img_b64 = payload["image_base64"]
    img_bytes = base64.b64decode(img_b64)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img_np = np.array(img)

    raw_result = predictor.predict_group(img_np)

    return safe_convert({
        "status": "ok",
        "record_id": payload.get("record_id", "unknown"),
        "result": raw_result
    })
