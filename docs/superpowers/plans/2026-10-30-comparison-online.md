# 模型比較新增研究候選與球季，發布GitHub及Streamlit

使用者授權更新TODO/PROGRESS/GitHub、模型比較介面加入原九項＋上季表現、1＋3＋同區＋上季表現＋2倍及2024–25，並更新線上。原九項也保留，共三個研究候選。

在comparison頁上方新增球季／比較範圍：2025–26預設原有模型，仍可切研究三種；2024–25只提供已驗證的研究三種。研究頁的三版本同樣本固定比較，與部署主模型分開；2025研究只CV9023/Feb3462，2024研究CV8647/Feb3634/Test7060。沒有產生新的模型、推論或測試。

不修改被凍結研究記錄的nba_dashboard/data.py、views.py、streamlit_app.py；比較頁改呼叫新入口，可轉原render_comparison。新建小型自足展示資料包，保存已存在三模型預測及獨立核對指標、來源SHA與完整manifest。只從此包讀取，不依賴訓練runtime/joblib。不將研究名次標成鎖定主模型，不跨季混樣本、也不以25–26九項凍結模型代替同樣本重訓九項。

- [ ] TDD資料包schema／樣本日期／hash／三模型指標、缺檔／tamper防護；建立package build及loader。
- [ ] 新增球季／範圍控制及研究比較圖表、分數／高分、逐場CSV；AppTest覆蓋兩季／CV/Feb/Test與原有頁回歸。
- [ ] 全測試、獨立review、離線乾淨checkout驗證；更新TODO/PROGRESS/README。
- [ ] 在codex/分支提交本次UI與必要研究程式／證據（排除Word、一次性文件腳本、環境與認證），推送到使用者已授權的部署main，確認Streamlit線上新介面。不要假設HTTP200即部署更新成功。

Ruling: 使用者明確要求更新GitHub和線上，完成檢查後直接推送，不再詢問發布許可。保留目前未追蹤研究依賴及使用者檔案，不清理或搬移舊檔。以新分支與獨立新模組完成，保護舊研究SHA鏈。
