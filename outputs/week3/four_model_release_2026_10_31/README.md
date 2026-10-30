# 四模型比較發布驗證

2024–25新增25欄增減＋上季表現＋高分2倍，固定四次訓練；2025–26沿用既有結果。兩季均增加第四個研究候選，所有模型維持共同樣本。

本次刻意修改research_views.py，以支援四模型。25–26舊實驗的validation.json包含當時UI來源雜湊，因此原verify_offset_prior25.py是對當時完整來源快照的核對；本次沒有改寫舊雜湊。historical_ui/nba_dashboard/research_views.py逐位元保留舊來源。本次發布驗證改用 `python -m scripts.verify_four_model_release`，只對這一個已授權更新的UI來源使用該舊快照，其他凍結模型、來源、結果均必須與原雜湊一致；同時重新核對25–26模型分量、24–25完整獨立稽核、兩季比較包的來源與逐場預測。

完整201項測試通過；full_tests.log記錄結果。Streamlit AppTest涵蓋兩季/時段/高分切換，實際分鐘仍在實際得分左邊。正式主模型/經典頁的data.py、views.py、streamlit_app.py均不變。已於線上核對兩季全部／30+結果並下載24–25的277筆高分逐場CSV；記錄在online_verification.json及online_*.png。GitHub功能版本5a017a6已發布至main。乾淨提交副本201項測試及發布稽核通過，使用既有venv，並非重新安裝依賴。
