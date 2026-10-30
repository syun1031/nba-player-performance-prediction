# 2025–26均值增減與上季兩模型比較

使用者要求比較以前相對五場均值增減與新兩種上季表現模型。比較四個完整方案：20項增減未加權、20項增減30+2倍、14項直接得分上季方案、25項直接得分上季2倍。五場均值另列。所有模型參數固定，原共同訓練13221，Jan31 cutoff；CV/Feb用既有固定fold預測，三月後只推論缺少的兩個增減模型，0fit、不調參。

同樣本三期：CV9023、Feb3462、2026-03-01～04-12的6979。嚴格核對(player_id,game_id)／日期／labels與baseline，依實際得分切all、30+、under30等分箱。CV保留F1/F2/F3分層。四個跨方案配對（各增減對各prior），每期all/30+日期群組2000次區間，共24個，未作多重比較校正。較早開發資料與三月test已重複使用，不稱新獨立測試。

不把差異解釋為單一特徵／目標的因果效果：兩個新方案還改了特徵組合與上季資訊。增減最終分數max(0,pts_mean_5+delta)，delta可負，不設上限；權重沿用原始得分規則。測試特徵直接重用已核對的test25欄（去標籤），不重新構造或改規則。

- [x] TDD：對照鍵／日期／標籤／baseline與fold完全一致；拒絕不匹配或重複；增減component加回與nonnegative。
- [x] 預存0fit協議與4模型hash，重用CV/Feb與prior-test，只有兩個增減模型新增test推論；保存全精度／繁中逐場與分箱／AP／24區間／報告。
- [x] 唯讀核對兩增減模型逐筆components及四模型指標／control不變，完整測試與獨立review；只更新本地研究進度，不換模型、不更動線上比較頁。

Ruling: 使用者已清楚指定比較對象，直接執行；涵蓋原增減未加權與2倍，避免只挑較好的版本。三月test已使用且新請求再次指定比較，明示為追加研究。新增獨立資料夾outputs/offset_vs_prior_2025_26/，不覆寫舊成果、不push/deploy本次未要求的介面功能。

Completed: 0fit, three common periods, 196 tests passed; independent review clean; no UI, GitHub push or production change.
