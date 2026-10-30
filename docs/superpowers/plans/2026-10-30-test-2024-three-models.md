# 2024–25三模型三月至季末預測

使用者明確要求開啟三月後比賽。本輪一次評估2025-03-01～2025-04-13，使用三個February凍結模型（訓練至2025-01-31），不重訓、不調參、不增候選。舊封存ledger保留作當時快照，新成果另存outputs/season_test_2024_25_three_models/，新test_usage記錄此次開啟與評估一次，拒絕覆寫重跑。

同日期前歷史逐場更新，過往三月已完成比賽可供較晚三月比賽使用；本場與同日、未來結果不入特徵。仍以確實有上場且來源MIN>0為條件，個人、球隊、對手至少五場，三模型共用完整25欄可用樣本。上季固定2023–24，沒有上季資料保留NaN。先發為賽後快照，仍是日期隔離的假設性回測，不稱賽前資訊完整重建。

新建獨立test-only特徵入口，保持舊development日期保護。重建同季個人窗口、陣容、對手、同區H2H與官方ID上季接合；在推論前刪除所有目標label，另作當場／未來標籤擾動驗證。來源MIN0（即使PTS>0）不進個人歷史但仍包含於官方球隊總分。以7238個預留原始鍵核對覆蓋，保存每筆排除原因，不能只挑高分。

- [x] TDD test日期／嚴格前日、同日未來毒值、保留前次development拒絕三月、不讀本場統計、missing prior與H2H規則。
- [x] 預存協議與三模型hash，0fit推論三模型，保存共同test features、逐場CSV／繁中CSV／分箱及30+警示/AP、分月統計、12組日期配對區間。
- [x] 獨立核對特徵／模型重載／scores與hash，完整測試及fresh review，更新三月已消耗的最新狀態與TODO/PROGRESS/PRD/README；線上不變。

模型參數源自後續2025–26研究，因此這是較早球季的固定模型保留區間回測；不能稱完全獨立的前瞻部署證據。結果不可再被稱為尚未使用的新測試資料。

Ruling: 使用者已明確解除三月封存，直接執行。既有來源與研究不覆寫，只新增test-only模組和成果，保留前次未追蹤依賴，不commit/push/deploy。

Completed: 0fits, 7060 common predictions, 151 insufficient history + 27 zero minutes excluded; 189 tests and independent review clean; test consumed once.
