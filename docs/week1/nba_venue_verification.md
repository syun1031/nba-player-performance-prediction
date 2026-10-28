# 五場主客方向差異的 NBA 官方核對

核對日期：2026-10-21。已查閱 NBA.com 比賽頁及 NBA 官方 scorer's report 第一頁。未修改兩份來源 CSV，也未重新編碼合併資料的 is_home。

## 結果

五場均在中立場地舉行。統計用指定主客隊與實際是否在球隊主場是不同概念；不能把頁面先後順序、比分順序或共用的 @ 文字直接當成一般主場優勢。

| 日期 | 對戰 | 官方實際場地 | 官方報告的主客標示 | 官方證據 |
| --- | --- | --- | --- | --- |
| 2025-11-01 | DAL／DET | Arena CDMX，Mexico City | VISITOR DAL、HOME DET | [Gamebook 第 1 頁](https://statsdmz.nba.com/pdfs/20251101/20251101_DALDET_book.pdf)、[NBA 比賽頁](https://www.nba.com/game/dal-vs-det-0022500147) |
| 2025-12-13 | NYK／ORL | T-Mobile Arena，Las Vegas | 首頁未標 HOME／VISITOR，不依排版指定 | [Gamebook 第 1 頁](https://statsdmz.nba.com/pdfs/20251213/20251213_NYKORL_book.pdf)、[NBA 比賽頁](https://www.nba.com/game/nyk-vs-orl-0022501229) |
| 2025-12-13 | SAS／OKC | T-Mobile Arena，Las Vegas | 首頁未標 HOME／VISITOR，不依排版指定 | [Gamebook 第 1 頁](https://statsdmz.nba.com/pdfs/20251213/20251213_SASOKC_book.pdf)、[NBA 比賽頁](https://www.nba.com/game/sas-vs-okc-0022501230) |
| 2026-01-15 | MEM／ORL | Uber Arena，Berlin | VISITOR MEM、HOME ORL | [Gamebook 第 1 頁](https://statsdmz.nba.com/pdfs/20260115/20260115_MEMORL_book.pdf)、[NBA 比賽頁](https://www.nba.com/game/mem-vs-orl-0022500578/box-score) |
| 2026-01-18 | ORL／MEM | The O2 Arena，London | VISITOR ORL、HOME MEM | [Gamebook 第 1 頁](https://statsdmz.nba.com/pdfs/20260118/20260118_ORLMEM_book.pdf)、[NBA 比賽頁](https://www.nba.com/game/orl-vs-mem-0022500602) |

DAL／DET 為墨西哥城賽事；兩場十二月十三日為拉斯維加斯 NBA Cup 準決賽；一月兩場為柏林與倫敦海外賽。地點由官方報告直接確認，不將統計 HOME 標籤誤解為球隊自己的主場。

## 對既有合併結果的影響

查閱原始得分檔，這五場兩隊的 MATCHUP 都使用 @，不是只有原先標記的那一隊如此。這是中立場編碼的一致現象，不能單靠它宣稱某份來源主客方向錯誤。先前稱為「主客方向衝突」的五筆，較精確應稱「中立場與主客表示差異」。

原先五筆球隊差異涉及合併檔 53 筆正分鐘紀錄；若新增場地屬性，應對同一場兩隊都標記，不能只修原先被標記的球隊。game_id 繼續保留使用者指定的原 MATCHUP＋日期，不為了主客編碼更換身份鍵。

## 建議的後續調整

新增 `is_neutral_site`，將這五場兩隊標記為已確認中立場；其他場次需要依資料來源核對，不能因未列於本次清單就宣稱全部非中立。

在實作前確認 is_home 表示「官方指定主隊」或「實際在自己主場」。若採後者，這五場兩隊皆不具有一般主場值，可與 is_neutral_site 一起表達；若採前者，三場可用報告明確 HOME，兩場 Cup 需另找指定主隊證據。不可靜默混用兩種定義。

本次完成官方核對；沒有將 53 筆未知值直接猜填，也沒有改動模型特徵定義。此前 merge_validation.json 記錄的是合併時來源差異，保留作稽核歷程。

## 使用者同意後的執行結果

已新增中立場欄位並重建合併檔：本五場兩隊共 101 筆正分鐘紀錄 is_neutral_site=1、模型 is_home=0，不將官方指定 HOME 誤當一般主場優勢；報告明確 HOME 另存 designated_home_team，兩場 Cup 未標者不猜填。原始 MATCHUP／game_id 不變。

原 53 筆方向差異旗標保留，但 home_away_resolution=verified_neutral_site，不再因此排除接合資格。未核對場次的中立狀態留空；尚未聲稱完成全季場地核對。merge_validation.json 已更新為本次結果，先前筆數與未知值情況保留於本報告及進度歷程。
