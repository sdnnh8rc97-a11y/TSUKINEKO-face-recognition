🚀 TSUKINEKO Face Recognition — V3 Pipeline
（Cosine + SVM + KNN Ensemble / Auto-Train / Group-Photo Training）

這份 README 設計給使用者、同事、主管都能一看就懂你整個系統的強度。

📖 目錄

系統簡介

功能特色

資料夾架構

安裝＆環境設定

使用方式

STEP1：放入新照片

STEP2：執行 V3

STEP3：獲得辨識與自動訓練

Auto-Train 訓練規則

模型介紹

[版控策略（legacy）](#版控策略）

未來 Roadmap

🧠 系統簡介

TSUKINEKO Face Recognition 是一套 高速、可自動訓練、可團體照輸入 的臉部辨識系統。
使用 insightface + 三分類器（Cosine + SVM + KNN）ensemble。

V3 版本加入：

⭐ 團體照自動裁切＋分類

⭐ 高信心照片自動加入資料庫（Auto-Train）

⭐ retrain SVM/KNN 秒級完成

⭐ 完整 cache 加速流程

⭐ 一鍵執行 pipeline

⭐ 自動化模型持續成長（越跑越準）

✨ 功能特色
✔ Ensemble 三分類器

Cosine 相似度

SVM（linear）

KNN（cosine distance）

三者投票決定最終預測（更穩定）。

✔ 支援團體照 → 自動裁臉 → 自動分類

自動偵測所有人臉

左到右排序

自動裁切

自動投票分類

高信心照片自動加入資料庫

✔ Auto-Train（安全版）

若同時滿足：

模型	        閾值
Cosine	    ≥ 0.65
SVM	        ≥ 0.80
Ensemble	  非 Unknown

→ 才會加入訓練集。

避免錯誤訓練污染資料庫。

✔ 自動 retrain

新照片加入後自動：

更新 raw

更新 cache

retrain 三分類器

立即可用

✔ cache 加速

第一次建立 cache 會比較久
之後更新 → 只跑新增照片（瞬間完成）

🔧 安裝與環境設定
1. 安裝 requirements
pip install -r requirements.txt

2. InsightFace 模型會在程式中自動下載

使用 buffalo_l：

3D Landmark

Detection

Recognition

🧪 使用方式（V3）
STEP 1 — 放入訓練照片

路徑（Google Drive）：

face_DataSet/
   face_raw/
      人名1/
      人名2/
      ...


你只需把新自拍照、團體照截圖丟進 face_raw/人名/

STEP 2 — 執行 auto_train_group.py
python auto_train_group.py


它會自動跑：

✔ cache
✔ database
✔ retrain
✔ 團體照分類
✔ Auto-Train
✔ 自動 retrain

STEP 3 — 取得分類結果

執行後會顯示：

每張臉的識別結果

3 個分類器的 confidence

是否成功加入訓練

🔒 Auto-Train 訓練規則

防止誤學習（污染模型）

加入訓練的條件：

cosine_conf ≥ 0.65
svm_conf ≥ 0.80
final_pred != "Unknown"

🤖 模型介紹
模型	       功能
Cosine	   主投票核心，速度最快
SVM	       可分類邊界更精準
KNN	       防止 SVM 偏移的第二保障
Ensemble	 綜合三者最終決定
