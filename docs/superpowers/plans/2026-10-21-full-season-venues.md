# 全季場地核對 Implementation Plan

**Goal:** 完成 2025–26 例行賽快照全部 1,230 場的 NBA 官方中立場核對，填入合併資料，不根據未列例外就推定為非中立。

**Architecture:** 由既有 source_rows.json 的官方 game ID 確定範圍；讀 NBA.com 公開比賽頁的 __NEXT_DATA__，保存 isNeutral、arena、主客隊與來源時間。每場核對 game ID、聯盟日期及兩隊後才接受；全部成功後才建立 registry 並重建合併檔。

**Tech Stack:** Python urllib.request、concurrent.futures、json、csv、unittest。最多四條唯讀 HTTP 請求並行，成功結果快取；不擷取帳號資訊，不安裝套件。

## 工作

- [x] 小樣本驗證官方頁欄位可讀；原受限網路失敗後依權限流程取得連線授權。
- [x] 為已確認非中立場與缺少官方主隊的錯誤狀態增加失敗測試。
- [x] 更新合併器以支援完整官方場地 registry，十二項測試通過。
- [x] 保存 1,230 場的官方場地證據，失敗不得補零或發布完整狀態。
- [x] 核對全部 game ID、日期、兩隊、中立旗標型別與球館，再建立 verified_venues.json。
- [x] 重建合併檔；核對場地完整覆蓋、來源不變、26,568＋83 筆數及同場兩隊一致。
- [x] 同步資料說明、規格、TODO 與進度，記錄實際全季中立場數。

## 欄位政策

已確認中立場 is_neutral_site=1、is_home=0；已確認非中立場 is_neutral_site=0，is_home 由官方 homeTeam 決定。官方指定主隊另存 designated_home_team，原始 game_id、MATCHUP 與數值不變。網路或欄位驗證失敗不推定場地狀態。
