# 2024–25 增減＋上季＋高分2倍與四模型上線

> **For agentic workers:** Use superpowers:executing-plans, with TDD and final review.

**Goal:** 沿用25–26已完成的25欄增減模型，在24–25重現，並在兩季模型比較增加第四個候選。
**Architecture:** 保留舊實驗與模型雜湊；新增24–25專用模型/執行器/唯讀核對。擴充獨立dashboard比較包，線上只讀既有結果。
**Tech Stack:** Python, XGBoost, unittest, Streamlit, GitHub.
**Spec:** 使用者本次明確要求與既有時間切分/欄位/權重協議。

## Constraints / review focus
- 25欄，delta=實際得分−前五場均值；原得分30+訓練權重2倍，每折正規化；評估不加權。
- 24–25 train12700/CV8647/February3634/Test7060；只有四次fit，Test使用1/31模型，不另調參。
- 測試期已使用；日期隔離假設性回測、上季缺值與先發快照限制均揭露。
- 保留所有舊模型/實驗檔，原始主模型與經典頁不變；新研究模型兩季皆可查看/下載。
- 核對共同鍵、標籤、日期、直接模型對照、實際分鐘、得分增減組合與指標。

## Task 1: 24–25 replication
- [x] 新增24–25日期門檻測試，確認先紅再綠。
- [x] 專用模型、四次固定訓練、三期間預測與日期群組區間。
- [x] 唯讀核對原始權重、模型重載、完整樣本、指標與來源雜湊。

## Task 2: Four-model comparison
- [x] 新增兩季第四模型與樣本完整性測試，確認先紅再綠。
- [x] 擴充比較包與UI為四候選，保留分鐘位置及下載、研究限制。
- [x] 全套測試、獨立review、TODO/PROGRESS/README更新。

## Task 3: release
- [x] 精確白名單提交、合併main並push（已獲使用者授權）。
- [ ] 線上確認兩季第四模型、整體/高分與下載介面，記錄證據。

## Completion ledger
- Task 1 complete: 24–25 training guard test RED→GREEN, exactly4fits, independent audit passed.
- Task 2 complete: fourth-model test RED→GREEN, 201 suite passed; exact predicted values/minutes verified in AppTest. Fresh independent review found no blocking issues.
- Historical UI intentionally changed; exact old research_views.py retained and checked by verify_four_model_release. Old model/data/output hashes unchanged.
- Review minor deferred: release verifier checks fourth model per cell; reviewer independently additionally checked all old-model cells in six bundles. Future automatic audit can include these old-model cells too; present outputs confirmed equal.
