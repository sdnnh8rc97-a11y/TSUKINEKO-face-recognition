# TSUKINEKO-face-recognition
Version 3 — 自動化人臉辨識系統（Ensemble + Auto-Train）

版本：v3.0.0（2025/11/28）
本版本是 TSUKINEKO-face-recognition 的 正式主線版本，以可上線部署為核心目標，完成全面重構，並加入自動訓練功能。

本版本的核心功能
1. 三分類器 Ensemble（Cosine + SVM + KNN Distance）

Cosine Similarity（支援 center vectors）

SVM（Linear SVC）

KNN（向量距離計算）

三者投票（tie-breaker：SVM Confidence）

✔ 大幅提高中距離 / 團體照的辨識穩定度
✔ 可避免單一模型偏差造成誤判

2. 自動裁切 + 棧板式排序

團體照偵測所有人臉

自動依 X 座標排序（由左至右固定順序）

自動產生 bbox、embedding、信心值

3. Auto-Train：自動新增訓練資料

當三分類器一致通過門檻時，自動：

裁切人臉 → 存入 face_raw/{person}/auto_x.jpg

即刻產生 512 維向量 → face_emb_cache/{person}/auto_x.npy

不用重跑全部資料：直接更新該員的訓練集

Auto-Train 保證資料量越用越大、準度越來越高。

4. Auto-Retrain（瞬間更新模型）

不用重新跑 3000 張照片。

本版本的 retrain 採用：

已存在的 cache（.npy）

遞增式 retrain（SVM + KNN）

重新訓練只需 2～4 秒。

5. 完整模組化架構（可供第三方整合）
src/
  face_detector.py
  face_embedder.py
  face_classifier.py
  predict_ensemble.py
  predict_group.py
  train_ensemble.py
  train_incremental.py
  legacy/   ← 舊版備份


此架構可輕鬆嵌入：

後端 API

AppSheet webhook

Google Apps Script（GAS）

Colab notebook automation

models/（模型輸出）

本版本會輸出以下模型：

models/
  label_map.pkl
  svm.pkl
  knn.pkl
  centers.pkl
  threshold.json

適用場景

保全部門點名系統

團體照點名（自動裁切、自動排序）

自走式增強訓練（Auto-Train）

低成本部署（無需 GPU server）

舊版（V1 / V2）

已移至：

src/legacy/


包含早期版本的：

單 classifier 實作

舊版訓練流程

舊版 group predict

保留作為研究用途。
