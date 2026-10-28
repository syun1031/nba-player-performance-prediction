# 先發隊友類別編碼 Implementation Plan

**Goal:** 將現有先發隊友名單轉為可輸入模型的 multi-hot 數值，保留原始版本。

**Architecture:** 每位訓練集中出現的先發隊友各一欄，是否同場先發為 1／0；順序無關。不用姓名的整數編號當特徵。另記未知隊友人數，類別名單只從合格訓練列建立並保存 JSON。

**Tech Stack:** Python 標準函式庫及 unittest。環境目前未安裝 Scikit-learn／SciPy，本步不為純編碼安裝套件；編碼器提供 fit、transform、fit_transform 與欄名介面，實際 Scikit-learn 訓練仍留第二週驗證。

**Spec:** docs/week1/integration_spec.md 的訓練集擬合規則與現有 team-lineup-v1。

## 工作

- [x] 寫失敗測試：順序無關、多位隊友、未知人數、未擬合拒絕、缺失／重複姓名拒絕、JSON 還原、重 fit 清除舊類別。
- [x] 實作 scripts/teammate_encoder.py，不用標籤或驗證／測試姓名建立類別。
- [x] 實作 scripts/encode_teammates.py，讀取歷史陣容合格樣本，輸出 outputs/week1/teammate_encoding/。
- [x] 輸出三集合數值 CSV、編碼器 JSON、欄位對照、共同樣本清單及未知覆蓋報告。
- [x] 核對類別等於訓練名單集合、欄位完全一致、已知旗標加未知人數等於四／五人、來源不變及還原後輸出相同。
- [x] 更新規格與 TODO，保留歷史先發為賽後資料的限制；不訓練模型、不評分測試。

## 使用限制

未知人數在訓練集必然為零，不能宣稱模型已學會未知球員的得分影響；它主要讓未知資訊可見並可追蹤。隊友姓名欄為可解讀對照，真正輸入是 0／1；欄位與編碼器必須跟模型版本一起保存。
