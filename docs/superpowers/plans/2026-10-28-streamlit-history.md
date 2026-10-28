# Streamlit 歷史回測與模型比較 Implementation Plan

**Goal:** 建立可本機操作的兩頁 Streamlit，展示既有最終模型與研究結果，不變更原模型與評估。

**Architecture:** nba_dashboard/data.py 純資料驗證與查詢，views.py 為 Streamlit／Altair 介面，streamlit_app.py 與 dashboard_pages/ 負責導航。前處理只作展示整理，不擬合、不預測。

**Tech Stack:** 既有 Python 3.12 .venv，新增 Streamlit 及其 Pandas／Altair 相依，保留 requirements-modeling.txt 所列版本。

**Spec:** docs/superpowers/specs/2026-10-28-streamlit-history-design.md。

- [x] 任務1：失敗測試後實作資料層；核對凍結成果、同樣本分組、前日／正分鐘近期紀錄與空資料處理。
- [x] 任務2：實作兩頁導航與繁體中文圖表／卡片／表格／下載；以 AppTest 測試實際互動與跨頁。
- [x] 任務3：啟動本機服務，確認可連線、檢視兩頁，完成獨立審查與來源不變核對。
- [x] 任務4：保存 requirements.txt、README 與介面驗證紀錄，更新 TODO／PROGRESS／PRD。

完成證據：資料層10項、AppTest6項，加既有75項，全套91項通過；pip check通過。模組／入口缺失、無效日曆日期與自洽但非鎖定版本的二月替換資料均先觀察RED，再實作GREEN。

Ruling: Streamlit AppTest的預設暫存目錄遭沙箱拒絕，依權限流程在核准後執行，未以其他方式繞過限制。本機健康端點也在核准後取得200。

Ruling: 窄視窗下原生top navigation會折到側欄，改用兩個可見page_link及hidden router；互動測試與實際導覽均通過。

Final review: 獨立審查的二月版本綁定Important已修正：證據／預測／指標三檔比對最終測試凍結來源雜湊；回歸測試含快取版本變動，複核無待修事項。原建模套件版本、凍結模型、預測及證據雜湊均未改變。

驗收紀錄及截圖：outputs/week3/streamlit_qa/validation.json、history.png、comparison.png。本機服務127.0.0.1:8501已開啟；瀏覽器交付頁已保留。未發佈或部署雲端。

Ruling: 專案目前無 Git 儲存庫，不建立工作樹或自行初始化 Git；不發佈、不部署、不啟用未來預測。安裝 Streamlit 時使用原建模版本清單作限制，避免任意更換既有相依。

## Review Focus

1. 切換比賽／球員後是否仍顯示舊選項或錯誤紀錄。
2. 最近十場是否混入當場、同日、之後或零分鐘紀錄。
3. 切換驗證／測試與高分分組時是否用到不同樣本或錯誤分母。
4. 主模型身份是否被測試排序或特徵切換改掉。
5. 來源缺失／改動、空選項時是否停止且不顯示過期的成功結果。
