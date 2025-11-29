# app.py — Cloud Run / FastAPI 入口

from fastapi import FastAPI
from pydantic import BaseModel
import base64
import cv2
import numpy as np
import sys
from pathlib import Path

# --------------------------------------------------
# 1. 把 src 加到 sys.path，方便匯入 v3.auto_train_group
# --------------------------------------------------
ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
sys.path.append(str(SRC_DIR))

from v3.auto_train_group import predict_single, predict_group  # 你剛加入的函式

app = FastAPI()


# --------------------------------------------------
# 2. Request Body 定義
# --------------------------------------------------
class SinglePayload(BaseModel):
    image_base64: str
    record_id: str | None = None  # 之後 AppSheet 可以丟任意字串來


class GroupPayload(BaseModel):
    image_base64: str
    record_id: str | None = None


# --------------------------------------------------
# 3. Base64 → OpenCV 圖片
# --------------------------------------------------
def decode_image(b64: str):
    data = base64.b64decode(b64)
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


# --------------------------------------------------
# 4. 健康檢查用（Cloud Run / 手動測試）
# --------------------------------------------------
@app.get("/")
def root():
    return {"status": "ok", "message": "TSUKINEKO face API running"}


# --------------------------------------------------
# 5. 單張臉辨識 API
# --------------------------------------------------
@app.post("/predict")
async def predict_endpoint(payload: SinglePayload):
    img = decode_image(payload.image_base64)
    result = predict_single(img)  # 呼叫你 V3.1 的邏輯

    return {
        "status": "ok",
        "record_id": payload.record_id,
        "result": result,
    }


# --------------------------------------------------
# 6. 團體照辨識 API（如果未來要用）
# --------------------------------------------------
@app.post("/predict_group")
async def predict_group_endpoint(payload: GroupPayload):
    img = decode_image(payload.image_base64)
    results = predict_group(img)

    return {
        "status": "ok",
        "record_id": payload.record_id,
        "results": results,
    }

# --------------------------------------------------
# 7. 啟動 API Server（Cloud Run / Colab 都能用）
# --------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8080,
        reload=False
    )
