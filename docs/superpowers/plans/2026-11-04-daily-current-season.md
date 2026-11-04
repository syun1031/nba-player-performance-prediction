# Daily current-season update implementation plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 每天台灣17:00更新2026–27歷史與本季假設性回測並發布Streamlit。
**Architecture:** 官方擷取 → 驗證歷史包 → 固定模型歷史推論 → 兩包同commit → 線上讀取。
**Tech Stack:** Python 3.12、GitHub Actions、Pandas、既有XGBoost/joblib、Streamlit。
**Spec:** docs/superpowers/specs/2026-11-04-daily-current-season-design.md（使用者已同意）。

## Global constraints

- 不重訓，不修改舊兩季凍結成果；嚴格前日歷史、官方ID、五場門檻。
- 本季歷史模式明確historical_postgame，不借用賽前時間檢查或偽造Confirmed。
- 失敗不發布，兩包同commit；無強制推送；保留使用者未追蹤檔案。

## Review focus

- 賽後目標數值或後續資料不能進入特徵。
- pending較早比賽、零分鐘與空合格資料不能默默通關。
- 每日發布須保留足夠來源證據，兩包必須同版。
- GitHub runner連線及套件安裝必須實跑，不能用本機測試代替。
- 三季切換保留正確widget身分及舊兩季原有結果。

### Task 1: 本季歷史特徵及資料包

Files: 新增 nba_dashboard/current_season.py、scripts/build_current_season_backtest.py、tests/test_current_season.py。
Interfaces: historical_features(bundle,target) → eligibility/25 features/audit；build(root,output) → verified artifact；load_current(root) → frames/manifest。
- [x] 先寫測試：五場邊界、同日/未來與目標結果隔離、ID、pending、分流與bundle驗證。
- [x] 跑測試確認RED；實作歷史模式並跑至GREEN。
- [x] 以本機現有歷史包產生初版，驗證模型SHA與舊成果不變。

### Task 2: 歷史回測頁新增本季

Files: 修改 nba_dashboard/routing_views.py；新增 nba_dashboard/current_views.py、tests/test_current_streamlit.py。
Interfaces: 球季2026–27分派render_current_history()，舊分支不更改數值。
- [x] 測試本季選單、分鐘位置、曲線/下載、空資料及三季切換，先RED。
- [x] 實作並跑本季/既有介面測試GREEN。

### Task 3: GitHub Actions每日發布

Files: .github/workflows/update-current-season.yml、scripts/update_current_season.py、tests/test_daily_update.py、docs/daily_update.md。
Interfaces: python -m scripts.update_current_season，以當次美東日期擷取並建兩包；Actions僅提交白名單outputs。
- [x] 測試日期、收集失敗拒用latest_success、雙包同版/安全發布，先RED。
- [x] 排程UTC09:00與workflow_dispatch、限同時一個、artifact、contents:write限定工作。
- [x] 全測試、路由稽核、fresh reviewer；更新TODO/PROGRESS。
- [x] 推送後手動執行Actions，確認runner成功與線上本季畫面。

## Ledger

- 2026-11-04：使用者已批准設計及發布。沿用本機既有.venv；工作分支codex/daily-current-season，保留所有既有未追蹤檔案。

- 2026-11-04驗收：271項本機測試通過；舊兩季38,805筆稽核通過；457筆新回測重建五場均值與兩固定模型推論一致。fresh reviewer發現空資料測試依賴正式包，已改固定非空fixture並保留empty測試，複查通過。
- GitHub Actions run 41796731502成功，包含安裝、官網擷取、兩包建立、稽核、自動提交與artifact保存。資料commit f202675，history_version=20261104T080537901773Z、through=2026-11-04，107場完賽、457筆合格。
- 線上已驗證2026–27球季、2026-11-04 16:06台灣更新時間及分鐘在得分左側；證據 outputs/daily_update_release_2026_11_04/。
