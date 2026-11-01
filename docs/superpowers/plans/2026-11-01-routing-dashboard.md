# 25分切換回測與兩季介面 Implementation Plan

> **For agentic workers:** Execute natively with TDD, independent final review and verification before release.

**Goal:** 依使用者固定規則改歷史回測預設，補24–25並更新比較頁及線上。
**Architecture:** 新增routing_data/routing_views/routing_comparison及獨立dashboard_routing資料包。沿用已凍結兩季四模型預測，不重訓、不重挑門檻；原data.py/views.py/research_data.py/research_views.py與dashboard_comparison均保留不改，避免破壞研究來源雜湊。
**Tech Stack:** Python/Pandas/Streamlit/Altair/unittest/GitHub.
**Spec:** 本次使用者指定近5場均分<25用直接25項＋上季＋2倍；>=25用五場均值增減25項＋上季＋2倍，兩季歷史頁與比較頁更新。按既有授權同步GitHub/Streamlit。

## Constraints / review focus
- 門檻25含等號，僅依pts_mean_5；本場實際得分/分鐘不影響分流，25項已含上季。不是上一場增減或方向懲罰模型。
- 兩季CV/二月/三月後共6份相同樣本，原四模型預測逐位元保留；新增route25與選用模型欄。
- 原始完整同季歷史用於近期10場，嚴格前日、原MIN>0，不只取有預測資格場次；實際分鐘在實際得分左側。每個五場均值獨立由歷史重建。
- 檔案hash/期間/鍵/分鐘/門檻/分流/指標不符時停止展示，不能用實際label決定路由。
- 切換球季/期間/球員不能沿用不存在的game/player選項。歷史頁選用模型、前五場均值、兩分支預測及CSV皆一致。
- 比較頁新增切換方案，保留四單模型及舊版演算法，預設顯示目前切換方案。兩季test均已使用，不稱新獨立最終測試。
- 這次是使用者指定展示策略，不宣稱route25穩定優於單模型。歷史先發仍賽後快照，日期隔離假設性回測。未來預測尚未上線。
- 不擅自覆寫/提交本機舊研究、Word或一次性腳本。白名單提交僅此介面/包/文件/測試；保存原來源hash。

## Tasks
- [x] 1. 純分流/邊界/label隔離、載入驗證/近期歷史/兩季六期間測試RED→GREEN；生成獨立比較包。
- [x] 2. 更新歷史與比較入口、AppTest涵蓋兩季/期間/球員/分支/分鐘/下載/缺檔，原單模型指標不變。
- [x] 3. 全套測試、獨立review、乾淨提交副本驗證、更新README/TODO/PROGRESS，白名單提交推送main。
- [x] 4. 實際線上核對兩季路由/比較/CSV，保存證據並更新交付紀錄。

## Completion ledger
- Tasks1/2 complete: boundary/fake-label test RED→GREEN; data/history validation and AppTest for both branches/seasons passed. Original expectations migrated to new default; archivedclassic checks retained.
- Full local suite220 passed. Independent audit38,805rows and fresh review passed; reviewer additionally verified18cross-context switches.
- No code/model changes in old frozen core/research data. Current release publishes only new routing modules/display bundle; prior untracked direction experiments remain local.

- Tasks3/4 complete:functionalcd026bd merged/pushed main; clean staged checkout207tests +audit pass with existingvenv; live two-season high/low branch values, twohistory CSVs and comparison CSV checked, screenshots recorded. No remaining findings.
