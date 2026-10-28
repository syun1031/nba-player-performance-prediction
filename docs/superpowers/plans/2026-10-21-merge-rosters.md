# 合併逐場數據與陣容 Implementation Plan

**Goal:** 依使用者授權合併兩份既有 CSV，不改來源；排除個人零分鐘樣本，完整保留隊友名單。

**Architecture:** 使用 Python 標準函式庫。姓名採七組明確對照，以日期、TEAM、對手與姓名核對；輸出 game_id 沿用得分檔 MATCHUP＋日期。方向衝突不猜測，is_home 留空；未知上場狀態不補零。

**Tech Stack:** Python csv、json、hashlib、unittest。

## 檔案

- `scripts/merge_game_rosters.py`：接合、輸出與驗證摘要。
- `tests/test_merge_game_rosters.py`：合併與錯誤處理案例。
- `outputs/week1/model_data/`：合併 CSV、排除清單、對照表、方向衝突表及驗證 JSON。

## 執行與驗收

- [x] 寫測試並確認未實作時失敗：姓名對照、零分鐘、先發／替補、方向衝突與非零分鐘無名單。
- [x] 實作 merge_records：必要欄位驗證、五名先發、唯一鍵、對手一致；正分鐘缺名單中止。
- [x] 執行測試，重現真實資料接合，核對 26,651＝26,568＋83。
- [x] 重新讀取輸出核對原數值、先發與隊友名單；驗證來源 SHA-256 不變。
- [x] 更新 TODO、PROGRESS 與來源核對報告；特徵窗口與模型工作仍未完成。

姓名對照是本快照限定的明確規則，不作跨來源模糊匹配；輸出保存對照清單。資料未包含賽前公布時間，不把歷史上場名單說成賽前已知資訊。本次不計算平常先發缺席與 rolling 特徵。
