# 全季 NBA 官方場地核對結果

核對日期：2026-10-21。已逐場核對保存資料內全部 **1,230 場**例行賽；中立場 5 場、非中立場 1225 場。不是以「未列例外」就推定非中立。

## 方法與證據

- 1227 場：從 NBA.com 公開比賽頁的 __NEXT_DATA__ 讀取 isNeutral、arena、homeTeam／awayTeam 與 gameEt，逐場驗證官方 game ID、日期與雙方球隊。
- 3 場：官方頁資料不可讀，改用官方 scorer's report 的球館與 HOME／VISITOR，並對照同主隊其他官方非中立場的同名主場球館。
- 三場補件為 2025-11-19 WAS／MIN、DEN／NOP、SAC／OKC；PDF 與文字保存於 pdfs/。它們的非中立判定依實際球館比對，不只依 HOME 文字。
- 只涵蓋已保存的 2025–26 例行賽集合，不混入季前賽、季後賽或 NBA Cup 決賽。

## 中立場清單

| 日期 | 兩隊 | 官方球館 | 官方來源 |
| --- | --- | --- | --- |
| 2025-11-01 | DAL / DET | Arena CDMX, Mexico City, MX | [NBA.com](https://www.nba.com/game/dal-vs-det-0022500147) |
| 2026-01-15 | MEM / ORL | Uber Arena, Berlin, DE | [NBA.com](https://www.nba.com/game/mem-vs-orl-0022500578) |
| 2026-01-18 | MEM / ORL | The O2 Arena, London, GB | [NBA.com](https://www.nba.com/game/mem-vs-orl-0022500602) |
| 2025-12-13 | NYK / ORL | T-Mobile Arena, Las Vegas, US | [NBA.com](https://www.nba.com/game/nyk-vs-orl-0022501229) |
| 2025-12-13 | OKC / SAS | T-Mobile Arena, Las Vegas, US | [NBA.com](https://www.nba.com/game/okc-vs-sas-0022501230) |

## 合併資料驗收

- 主合併檔 26,568 筆，is_neutral_site=1 為 101 筆、=0 為 26467 筆，無空值。
- is_home 在中立場兩隊皆為 0；非中立場依官方指定主隊決定。官方 designated_home_team 與球館另存。
- 原 MATCHUP、game_id、原九欄數值不變；83 筆零分鐘仍另存排除清單；兩份來源 SHA-256 不變。
- 原五筆方向差異的 53 筆主檔紀錄保留稽核旗標，已由官方場地證據處理。
- 場地完整不代表五場歷史資格、陣容賽前公布時間或模型已完成。

## 重現

核對結果：validation.json；逐場欄位：official_game_venues.csv；供合併器讀取：verified_venues.json；每場保存欄位：games/<官方 game ID>.json。

先執行 scripts/fetch_official_venues.py（需網路，成功結果會快取）；三場缺頁以 scripts/supplement_venue_pdfs.py 補件；再執行 scripts/validate_official_venues.py 及 scripts/merge_game_rosters.py。原 HTTP 批次的錯誤紀錄保留在 fetch_summary.json，三場後續補件後的完整狀態以 validation.json 為準。

以上是賽後取得的場地資料；不代表已重建當時賽程公布／更改時間。
