# 逐場數據與陣容合併結果

更新日期：2026-10-21。

## 檔案

- `nba_2025_26_player_games_merged.csv`：26,568 筆 MIN>0 的球員比賽紀錄，保留原九欄、原順序，加上陣容與接合欄位。
- `excluded_zero_minutes.csv`：83 筆 MIN=0，已從主合併檔排除；其中三筆不在陣容名單。
- `name_aliases.csv`：七組明確姓名對照，以得分檔 PLAYER 作輸出姓名，不做模糊匹配。
- `matchup_conflicts.csv`：保留五筆來源方向差異的稽核紀錄，影響主檔 53 筆；現已用官方中立場證據處理，不代表仍待猜填。
- `merge_validation.json`：來源 SHA-256、筆數、比對與重新讀檔驗證結果。

CSV 為 UTF-8 BOM、CRLF。名單欄位為 JSON 陣列，Python 可用 `json.loads()` 解析；不依陣列位置推定球員位置。

## 新增欄位

| 欄位 | 說明 |
| --- | --- |
| player_id／team_id | 沿用 PLAYER／TEAM，不是官方數字 ID |
| game_id | 得分檔原 MATCHUP＋`|`＋GAMEDATE，保留球隊視角 |
| is_starter | 有星號為 1，非先發為 0；排除清單中未匹配者留空，不假設替補 |
| starting_teammates | 本場實際先發集合排除自己；自己先發四人、替補五人 |
| appeared_players | 本隊本場完整實際上場名單，含零分鐘球員，姓名已按對照統一 |
| ROSTER_MATCHUP | 保留陣容來源原對戰文字 |
| roster_match_status | matched 或 missing_player |
| home_away_conflict | 原始兩來源對主客方向不一致為 1，即使已由中立場證據處理仍保留 |
| is_home | 模型的主場指標：已確認中立場兩隊皆為 0；其他場次保留一致來源的主場 1／客場 0；未解決差異才空值 |
| is_neutral_site | 全季已逐場核對：中立場為 1、非中立場為 0，無空值；共 5／1225 場 |
| designated_home_team | 官方報告明確標示的統計主隊；與模型 is_home 分開，未標者空值 |
| venue／venue_source_url | 全季已確認的場地與官方比賽頁／記錄表來源 |
| home_away_resolution | verified_neutral_site 或 verified_official_venue；原始方向差異旗標仍保留 |
| merge_eligible | 僅表示正分鐘、成功接合且方向差異已處理，不等於已滿五場歷史或完整模型資格 |
| exclusion_reason | zero_minutes、roster_player_missing、home_away_conflict；多原因以分號分隔 |
| pregame_verified | 全部 False：來源沒有先發／缺席資訊的歷史公布時間 |

全季 1,230 場已核對，5 場中立場涵蓋兩隊共 101 筆主檔紀錄，is_home 全為 0；其中 53 筆為原先來源方向差異。其餘 26467 筆 is_neutral_site=0，is_home 按官方主隊決定。全部 26,568 筆通過接合資格。後續已建立歷史窗口資格，見 [基礎特徵與基準](../features_baseline/README.md)；本檔的 merge_eligible 仍只表示接合資格。

逐場場地與來源見 [全季核對](../venue_audit/README.md) 與 official_game_venues.csv。

## 驗收與限制

原始數值逐欄重新讀檔比對一致，所有正分鐘紀錄有陣容對應，來源 SHA-256 不變。十二個單元測試涵蓋原七種情況與中立場兩隊處理、未核對場地不猜填、不同對手不套用覆寫，以及官方非中立場與缺少主隊時的驗證。

資料是歷史實際陣容；沒有賽前公布時間，不得稱為已驗證的賽前先發／缺席資訊。尚未計算 rolling 特徵、平常先發、隊友缺席或模型評估。未列於上場名單不等同賽前確認缺席。

## 重現

從專案根目錄，使用可用的 Python 3：

```powershell
python -m unittest discover -s tests -p test_merge_game_rosters.py
python scripts/merge_game_rosters.py
```

只使用標準函式庫，不需新增套件。腳本可重建本目錄合併結果，不能寫回兩份來源 CSV。
