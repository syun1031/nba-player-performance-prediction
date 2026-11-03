# Future Prediction Implementation Plan

> **For agentic workers:** 使用superpowers:executing-plans在目前session逐項執行，完成後做獨立審查。

**Goal:** 新增由NBA官方Confirmed先發解鎖的2026–27未來預測頁。

**Architecture:** 新增source、data、inference與views模組，既有研究資料／模型／欄位維持凍結。手動更新歷史包，網頁查詢當場賽程與正式名單，預測前重查來源。只載入repo內固定雜湊模型。

**Tech Stack:** 現有Python、Streamlit、Pandas、XGBoost、joblib、unittest；不安裝套件。

**Spec:** `docs/superpowers/specs/2026-11-03-future-prediction-design.md`。

## Global Constraints

NBA官方ID與Confirmed/Active、兩隊五先發、賽前30分鐘內、嚴格前日同季、個人／兩隊五場門檻；API無時區timestamp只保存原文。採固定25分分流。歷史包與固定模型須驗證雜湊。只下載推論紀錄，不接受外部模型上傳。來源失敗一律顯示等待／錯誤並停用。

## Review Focus

來源日期／身份／時間狀態互相矛盾；缺失或預估先發；歷史包未更新但顯示最新；午夜／T-30／開賽界線；換球員或重查後殘留舊結果。

## Tasks

- [x] TDD官方來源解析與驗證，實測保存Confirmed歷史結構及未公布404；先紅再綠。
  - `nba_dashboard/future_source.py`：查詢／解析每日賽程、逐場資料與daily_lineups；只允許官方URL；時間及先發狀態。
  - `tests/test_future_source.py`：兩隊、ID、Confirmed、Inactive、時間、404、Naive timestamp不推論。
- [x] TDD資料包、25項特徵與固定模型推論，更新本季歷史至11/3，打包至`outputs/future_prediction`。
  - `nba_dashboard/future_data.py`：只讀資料／模型雜湊、歷史資格與前日覆蓋檢查。
  - `nba_dashboard/future_inference.py`：沿用共用特徵，泛化本季h2h與上季profile，固定模型分流與下載紀錄。
  - `scripts/build_future_prediction_bundle.py`：僅打包既有成功快照，不擬合。
  - `tests/test_future_inference.py`：五場門檻、同日／未來隔離、25項順序、兩分支、凍結模型重現、分鐘精度。
- [x] TDD Streamlit未來頁、三頁導覽、查詢／等待／錯誤／推論與下載；保存原入口供舊證據核對。
  - `dashboard_pages/future.py`、`nba_dashboard/future_views.py`、`streamlit_app.py`。
  - `tests/test_future_streamlit.py`及既有導覽測試。
- [x] 完整單元／AppTest、歷史38805筆獨立稽核、真人來源實測、瀏覽器等待畫面與下載確認；獨立審查。
- [ ] 更新README／TODO／PROGRESS；明列尚未實際觀測到賽前Confirmed成功例，而不是把歷史名單變成賽前證據；提交GitHub與驗收線上版。

## Ledger

- 2026-11-03：使用者已選擇官方自動先發驗證，無人工替代。Builder check回覆nothing important left。
- 官方daily_lineups_20261102.json成功，含Confirmed/Active/position/personId/teamId；timestamp無時區。20261104尚未公布404。11/4逐場頁狀態1且players空，不能以此產生先發。
- 19筆可對上姓名的2025–26官方分秒與既有MIN抽樣皆符合四捨五入整數分鐘（30秒進位）。保留原秒數作顯示，模型MIN用同樣規則；模型MIN為0者不進歷史窗口，符合原本訓練來源零分鐘排除。
- 本次使用者明確要求先實作官方取得與驗證，因此按已選流程實作，不再要求重複許可；來源尚未通過的狀態維持等待。

- 2026-11-03完成：22項future tests、完整253、乾淨輸出240與原38805筆回測audit均通過。獨立review兩項修正通過。真實官網11/4有14場賽程，先發未公布404，瀏覽器等待畫面已保存；沒有實際賽前Confirmed成功案例。
