# 未來預測使用方式

2026-11-03新增。線上入口：[未來預測](https://nba-player-performance-prediction1.streamlit.app/future)。

1. 選擇NBA比賽日期（美東日期），按「查詢官網賽程」。開賽時間會轉成台灣時間。
2. 選比賽，按「查詢官網正式先發」。只接受NBA官方daily lineups明確Confirmed、Active的名單；不手動推定五人，也不從過去比賽抄先發。
3. 兩隊各五位先發確認後，可選啟用名單內的先發或替補。個人、球隊與對手均至少五場同季前日歷史，資料沒有未釐清比賽，才有預測資格。
4. 開賽前30分鐘內、尚未到開賽時間，按「產生預測」。系統會重新核對官網名單與賽程，再使用固定模型推論。若先發、啟用或時間狀態變更，停用本次預測。
5. 下載JSON完整紀錄或CSV摘要留存。兩種檔案都記錄來源網址、實際取得UTC、模型雜湊及歷史場數；JSON另保留來源內容。雲端暫存不是永久紀錄。

目前使用原本訓練截至2026-01-31的兩個模型，沒有新增訓練；本季前瞻準確率仍待累積。近五場平均<25用直接25項，>=25用五場均值增減25項。25項已包含上季5項表現，高分2倍為訓練時的權重，不是在網頁把預測乘兩倍。

## 目前為何可能無法預測

- 官網尚未公布、仍是Projected、兩隊任一不足五位先發、球員Inactive：等待確認。
- 距開賽超過30分鐘：可先看資料；尚未進入正式預測時段。
- 已到開賽時間或官網顯示進行中／完賽：停止產生新的賽前預測。
- 歷史尚未查詢至目標日期前一天、較早日期有未釐清比賽、五場不足：先更新或等資料累積。
- 來源連線或格式錯誤：顯示原因，不能使用前一次名單繼續生成新的數值。

官網原始timestamp沒有時區，僅保存原文字；不假設它是UTC，也不把擷取時間冒充發布時間。是否先發必須在本次實際取得且還沒開賽時確認。

## 首次發布的資料限制

目前包查詢至NBA日期2026-11-03，收錄101場、2,279筆球員與202筆球隊紀錄，582份上季表現以NBA球員ID核對。本次來源另外有6場狀態待釐清（10/27、10/28、10/30、11/2），其中部分在11/2快照曾是Final，但11/3每日賽程與逐場頁卻回到未開始或進行中。沒有擅自把舊結果覆蓋成新版本。這些較早比賽尚未解決前，頁面會阻擋數值預測。

已成功取得官方Confirmed欄位結構，但當時為進行中或完賽資料；11/4先發檔尚未公布404。已驗收真實官網等待流程；尚未實際觀測一筆在開賽前取得Confirmed並完成預測的成功案例。單元與AppTest成功案例是明確標示的測試資料，不是正式預測。

個人分鐘特徵會依舊訓練資料規格轉為整數分鐘（30秒進位），四捨五入為0者不計模型個人歷史；近十場表格保留原秒數。已用官方2025–26單場19名可對上姓名者驗證轉換。上季每分鐘得分仍沿用原訓練來源的分鐘精度。

## 每日手動更新歷史包

在專案根目錄的PowerShell執行：

```powershell
.venv/Scripts/python.exe -X utf8 -m scripts.fetch_current_season
$nbaLatest = Get-Content outputs/nba_2026_27/latest_success.json -Raw | ConvertFrom-Json
.venv/Scripts/python.exe -X utf8 -m scripts.build_future_prediction_bundle --snapshot "outputs/nba_2026_27/$($nbaLatest.snapshot)" --replace
```

擷取不成功時先處理錯誤，勿繼續使用舊latest_success當作本次更新。打包會驗證來源雜湊、球員／比分／分區與模型；成功後才切換新包。`--replace`保留前一包於`outputs/future_prediction_archives/`，不刪原始快照。來源中尚有pending或矛盾比賽時會保留該狀態，並不因此聲稱歷史完整。

打包成功後，將`outputs/future_prediction/`的新版本提交到GitHub，線上版才會更新歷史；官網賽程與先發則由頁面按鈕即時查詢。不設定自動排程。

## 驗證

```powershell
.venv/Scripts/python.exe -X utf8 -m unittest tests.test_future_source tests.test_future_inference tests.test_future_streamlit
.venv/Scripts/python.exe -X utf8 -m scripts.verify_routing_dashboard
```

此功能沒有盤口、賠率或ROI，也不把預測分數視為投注勝率。
