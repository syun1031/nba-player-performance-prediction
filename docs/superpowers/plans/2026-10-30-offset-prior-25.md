# 25項增減＋上季表現＋高分2倍實驗

使用者明確要求測試，新增唯一候選：既有1＋3＋同區20欄加上季5欄，目標target_pts - pts_mean_5；預測max(0,pts_mean_5+delta)。增減可負、不設上限，訓練權重依原始PTS>=30給2倍後以本折平均正規化，不依delta判斷。

2025–26固定原XGBoost參數，同train13221、CV9023、Feb3462；F1/F2/F3和January31共4fit，不調參。對照原20項增減2倍與25項直接得分2倍（已有凍結成果，不重訓）。第一對只增加上季欄，第二對相同25欄與權重但預測目標不同。

嚴格重用已核對的v9 train/validation與25季test features，不改缺值、姓名接合與歷史資格。三月6979已使用test，只列追加研究，不稱新的獨立證據；用Jan31凍結模型推論，不能用test結果再調參。先發為賽後快照，仍為日期隔離假設性回測。

- [x] TDD：原PTS權重、signed delta目標、非負只裁最後分數、25欄／缺值與推論不讀labels、非法日期／重複／非有限資料不消耗fit。
- [x] 預存4fit協議；保存三折及Jan31模型，重載相同、去掉／改動當場標籤不變；三期同樣本對照，全精度與繁中逐場、分箱／prior分組、警示/AP與12個日期區間（2對×3期×all/30+，2000次，不校正多重比較）。
- [x] 唯讀核對profile來源／模型components／weights／controls／scores與hash，完整測試及fresh reviewer；更新本地TODO/PROGRESS，不替換主模型、不自動推送或改線上UI。

另存outputs/offset_prior25_w2_2025_26/，拒絕覆寫。既有研究與未追蹤的前輪檔案全部保留，不重做前輪。沿用當前codex研究分支，不搬移共用資料。

Completed: exactly 4 fits, 199 tests and independent review clean. Existing controls and production unchanged; no push or deployment.
