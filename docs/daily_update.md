# 每日自動更新2026–27

GitHub Actions工作流程：`.github/workflows/update-current-season.yml`。

每天UTC09:00（台灣17:00）開始執行，GitHub可能延遲排程，17:00不是保證發布完成的時間。工作在GitHub執行，不需要本機開機。亦可至repo的Actions → Update 2026-27 data and backtest → Run workflow補跑。

流程：安裝鎖定套件 → 測試 → 以當次美東日期為截止擷取NBA官方例行賽資料 → 建立未來預測歷史包 → 固定模型產生本季假設性回測 → 驗證兩包與舊兩季成果 → 在同一個commit提交兩包 → Streamlit偵測main更新。

只有`outputs/future_prediction/`與`outputs/current_season_backtest/`自動提交。完整來源收據與診斷以當次Actions artifact保留30天，可下載延長留存。內建GITHUB_TOKEN只在此job要求contents:write，不需要PAT；不改全repo預設讀取權限。動作版本以GitHub官方release的commit SHA鎖定。

任何一步失敗就不提交，線上保留上一份完整資料。推送衝突不強制覆寫，請從Actions重新執行。官方來源若拒絕雲端連線，該輪明確失敗，不能用舊latest_success冒充新資料。賽季擷取視窗至2027-06-30，之後應停用或更新球季。

歷史回測頁新增2026–27，顯示開季至最新已完賽的合格球員出賽。個人、球隊、對手均至少5場，嚴格只讀目標日期前的歷史；先發取完賽快照，因此稱「日期隔離的假設性回測」，不當成真正賽前留存結果。較早日期仍有pending比賽則暫不回算受影響場次，排除原因保留於excluded.csv。零分鐘未納入，原始分鐘用於顯示，特徵分鐘沿用舊訓練規格。

沿用固定兩模型，不重訓：近5場均分<25用直接25項＋上季＋2倍，>=25用五場均值增減25項＋上季＋2倍。模型訓練截至2026-01-31，上季特徵對應2025–26。高分2倍是訓練權重，並非將預測加倍。2024–25及2025–26仍固定3月至季末，既有結果不改動。

未來預測頁使用同次歷史包，仍須當次官方Confirmed、Active、開賽前30分鐘與各五場資格。每日17:00更新不會自動產生隔日賽前預測；另訂的單場任務仍由原排程處理。

本機手動執行同一流程：

```powershell
.venv/Scripts/python.exe -X utf8 -m scripts.update_current_season
```

本機執行僅產生資料；GitHub Actions才包含自動提交。使用者應只提交上述兩個已驗證資料夾，避免加入其他研究或暫存檔案。

官方排程限制：https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#schedule
