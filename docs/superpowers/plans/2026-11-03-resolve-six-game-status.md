# Six Game Status Resolution Plan

> **For agentic workers:** 使用superpowers:executing-plans與systematic-debugging。使用者明確授權先處理六場，不改既有預測政策。

**Goal:** 逐場以官方證據釐清六場pending，更新歷史包並保留來源與原始狀態。

**Architecture:** 對過了開賽時間但狀態未Final的每日卡片補查官方box；只有完整Final證據與另一份官方Final來源一致才能衍生歷史完賽。未來先發驗證碼維持原樣。

**Tech Stack:** 既有Python收集／打包流程、unittest、CSV／JSON、Streamlit。

**Spec:** 使用者「先處理這6場」及`outputs/six_game_resolution_2026_11_03/README.md`的驗收規則。

- [x] 比對原快照、當前官方box、canonical頁及官方daily lineups；CDN／legacy端點403保留探查失敗，不據以判斷完賽。
- [x] 先紅後綠新增反例測試：Final狀態、時間、分鐘、比分、隊伍及先發ID與Confirmed等。
- [x] 完整重收107場；修正逐秒精度累積3秒的過嚴檢查，最多每名有效球員1秒且有官方team.minutes核對。
- [x] 獨立審查無重大問題。補回六場134筆，原2,279筆數值及身分完全相同；27筆原始分鐘格式改變但解析秒數相同。
- [x] 更新歷史包，pending=0；11/4的歷史覆蓋門檻通過，先發與時間門檻未修改。
- [ ] 完整回歸、原38805筆回測稽核、發布資料包與線上驗收。

## Ledger

四場box數字狀態碼仍為1/2，但Final文字、零秒、完整box及官方每日lineups狀態3和先發ID一致；後兩場已正常Final。沒有擅自解釋官網naive timestamp的時區，全部只記實際擷取UTC。

- 驗證：完整257項、乾淨基準同步本次12個發布檔案後244項測試通過；原38805筆稽核通過。資料只新增六場134筆，27筆分鐘原文格式差異解析後完全相同。
