# 最近出賽特徵與基準 Implementation Plan

> 在本 session 依 executing-plans 執行，不另分派工作；使用者已同意建立特徵與基準。

**Goal:** 從已合併、已核對場地的資料產生基礎特徵、共同樣本清單與開發期五場平均基準。

**Architecture:** 使用可重用的單筆預測函式，只接收此前日期的同球員同季紀錄；歷史資料與未來請求可共用。歷史批次先排零分鐘，再按日期建立特徵及切分；原 CSV 保持不變。

**Tech Stack:** Python 標準函式庫 csv、json、statistics、math、unittest。本週不訓練機器學習模型，不安裝套件。

**Spec:** docs/week1/integration_spec.md。

## 固定規則

- 先前至少五場才能預測；最多十場平均可以使用五至九場的實際分母。
- MIN=0 不當作樣本或個人歷史，MIN>0 且零分保留。
- 同球員跨隊延續、跨季重置；不使用同日與未來結果。
- 八個基礎輸入：pts_mean_5、pts_mean_10、min_mean_5、reb_mean_5、ast_mean_5、is_home、is_neutral_site、rest_days。
- 訓練 2025-10-21～2026-01-31，驗證二月，測試 2026-03-01～2026-04-12。
- 只算訓練／驗證 MAE、RMSE，測試只產生資料與資格清單，不輸出分數或基準預測。
- 本場實際先發與缺席不混入基礎模型；原合併檔仍保留陣容資訊。

## 檔案與介面

- scripts/recent_game_features.py：features_for_request(history, target)、build_feature_rows(records)、split_for_date(day)。
- scripts/build_features_baseline.py：讀合併資料、寫特徵／切分／manifest、指標與白話報告。
- tests/test_recent_game_features.py：因果時間、窗口與基準指標測試。
- outputs/week1/features_baseline/：輸出，不覆寫來源。

## 工作與驗收

- [x] 先寫測試並確認失敗：當場／未來修改不影響、零分鐘排除、零分保留、轉隊、跨季、同日、缺失窗口、切分邊界、指標。
- [x] 實作共用單筆與歷史批次特徵；缺失或非法窗口回傳原因，不跳過壞場補更早資料。
- [x] 實作輸出與基準，保存樣本鍵、窗口 ID、來源 SHA-256 與規則版本。
- [x] 核對八欄皆有限值、歷史最大日期早於目標、鍵唯一、切分不交錯、原始數據雜湊不變。
- [x] 從輸出重新計算開發期基準；測試分數不存在；對全季資料做未來值變動與順序變動的檢查。
- [x] 更新 TODO、PROGRESS、PRD、整合規格；陣容與未來來源工作仍維持未完成。

## 執行記錄

Ruling: 本步先做已確認的基礎數值模型，不加本場陣容 — 缺席的賽前證據尚未補齊 — 不影響之後在相同樣本比較陣容擴充。

Ruling: 日期不足或無法解析的輸入中止；數值缺失會留下明確原因，且不偷偷從歷史窗略過 — 避免資料問題被平均值掩蓋。

Ruling: 目前工作目錄為既有資料專案，保留所有檔案；不為本次工作建立新儲存庫或自動發布。

Review: 檢查發現 Python 也接受沒有連字號的日期，若直接按字串比較可能切錯集合；已先加入失敗測試，再要求 YYYY-MM-DD 格式。完整 24 項測試及實際資料驗證通過。輸出 23,722 筆合格資料，驗證 MAE 4.798；沒有查看測試誤差。
