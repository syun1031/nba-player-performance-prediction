# 第一週 NBA 資料來源調查

查證日期：2026-10-13（UTC+08:00）。範圍：全聯盟球員下一場得分、以出賽為前提；歷史回測及未來例行賽預測；手動更新及 Streamlit 展示。本報告只提供調查與流程提案，不代表來源選型、部署或歷史資料核對已完成。

已閱讀根目錄 PRD.md、TODO.md、PROGRESS.md。既有 CSV 的筆數、球員數與完整性仍待重新核對；本次未執行既有核對程式。候選方案共 **兩個**，NBA CDN 與 Stats 端點屬同一官方來源方案，不另算第三個。

## 建議與目前可用程度

- **研究與既有資料銜接：優先 NBA 官方來源 + nba_api 文件，採本機手動取得、驗證後產生版本快照。** 官方 ID 最容易與既有原始連結銜接，但本次三個官方端點均未成功取得 JSON，現在不能宣稱此環境可以穩定更新。
- **若需較明確的公開 Demo 使用條件及 API 介面：BALLDONTLIE ALL-STAR 是備選。** 逐場統計、現役球員與賽程在此級別可取得；需後續決定預算、帳號及 ID 對照。本次沒有帳號、API key 或付費端點實測。
- Streamlit 優先讀取已驗證的持久快照；資料採集不放在每次頁面重跑。NBA 連線失敗或使用條件未收斂前，不應把官方方案列為已可公開部署。以上是技術選型建議，尚未採購或發布。

## 方案比較

| 面向 | A：NBA 官方 Stats/CDN，nba_api 為候選客戶端 | B：BALLDONTLIE NBA API |
|---|---|---|
| 歷史逐場 | PlayerGameLogs：PLAYER_ID、GAME_ID、TEAM_ID、GAME_DATE、MATCHUP、MIN、PTS、FG_PCT、REB、AST；Season、SeasonType、日期可篩選 [A1] | `/v1/stats`：player.id、game.id、team.id、min、pts、fg_pct、reb、ast；支援球季／日期／球員／比賽篩選及游標分頁 [B1] |
| 未來賽程 | ScheduleLeagueV2：gameId、主客隊 ID、gameDateTimeUTC、狀態、延賽欄位；CDN 為另一公開網址候選 [A3] | `/v1/games`：id、datetime、home_team、visitor_team、status_state、postponed；日期範圍篩選 [B1] |
| 球員 ID／目前球隊 | CommonAllPlayers：PERSON_ID、TEAM_ID、ROSTERSTATUS；CommonTeamRoster 可按球隊取得 PLAYER_ID／TeamID，適合交叉確認 [A2,A4] | `/v1/players/active`：id、姓名、team；`/v1/players` 可查歷史球員，但不能把所有球員都當現役 [B1] |
| 取得方式 | HTTPS GET；可用 nba_api 或直接 HTTP；無需在本次安裝客戶端。Season 例：2025-26，未來候選：2026-27。nba_api 是社群專案，並非 NBA 官方 SDK [A5] | HTTPS GET，Authorization 放 API key；`per_page` 最大 100，追蹤 next_cursor 至結束。使用服務自身 ID，不等於 NBA ID [B1] |
| 覆蓋與品質 | 文件確認可查球季，但本次未驗證 2025-26 全季或其他年份完整性。部分端點文件 last_validated 為 2020，賽程文件為 2025，須核對實際 schema [A1–A4] | 文件標示 1946 至今；這是服務整體涵蓋宣告，不能推論各年份所有逐場欄位都完整。2025-26 的實際涵蓋仍待測 [B1] |
| 更新頻率 | 查閱資料未提供可依賴的固定發布間隔、配額或 SLA；賽程與名單應每次手動更新重取，完成比賽仍可能更正。這是流程建議，不是供應商頻率保證 | 文件稱比賽中的球員統計即時更新；賽程／轉隊同步及賽後更正的延遲未確認，不視為 SLA [B1] |
| 限制與費用 | 公開 URL 無 API key；未找到正式公開價格／配額承諾。nba_api MIT 授權適用客戶端程式，不能當成資料授權。403、逾時與 schema 變動需處理 [A5] | Free：$0、5 次/分鐘；ALL-STAR：US$9.99/月、60 次/分鐘；GOAT：US$39.99/月、600 次/分鐘。Free 不含逐場統計與現役名單，本專案不需為高階端點選 GOAT。價格以查證日文件為準 [B1,B2] |
| 使用條件 | 需標註 NBA.com；非商用不等於可公開完整更新資料庫，該用途需事先同意 [A6]。本專案公開資料／Demo 是否符合條件尚待確認 | 條款允許合法取得資料的分析、快取、衍生作品與 ML；限制競爭服務、原樣轉售、共享 key 及規避配額。不得稱資料為官方聯盟資料 [B3] |
| Streamlit 適用性 | 可讀本機產生的快照；本機連通不代表雲端 IP 連通，且本次本機也未取得 JSON。不建議前端使用者觸發大量官方查詢 | 適合伺服器 HTTP 或本機更新快照；若雲端取資料，key 放 Streamlit secrets，不提交 GitHub。雲端連線與付費資料品質均待測 [S1] |

A1、A2 等連結見文末，皆為 NBA 網站或候選工具／供應商自身文件。nba_api 靜態球員清單只有 ID、姓名與 is_active 等欄位，沒有目前球隊，且隨套件更新；不能代替動態名單 [A7]。

## 欄位與識別碼設計提案

| 資料表 | 最小保留欄位 | 用途與防錯 |
|---|---|---|
| historical_player_games | source、player_id、game_id、game_team_id、game_date_source、season、season_type、MIN、PTS、FG_PCT、REB、AST | 主鍵 `(source, player_id, game_id)`；保留該場球隊，禁止用目前球隊覆蓋歷史 |
| schedule | source、game_id、home_team_id、away_team_id、scheduled_start_utc、source_date、status、postponed、season、season_type | 保留原始狀態／日期；同場改時間以 game_id 更新，不按日期製造新場次 |
| roster_snapshot | source、player_id、display_name、current_team_id、roster_status、observed_at_utc | 「目前」只代表採集時快照；轉隊同步延遲與自由球員／空 team_id 需處理，不代表確認出賽 |
| manifest | snapshot_id、來源 URL／參數、schema_version、fetched_at_utc、validated_at_utc、資料涵蓋範圍、各表筆數、hash | 對 UI 提供版本與時間；請求失敗不更改成功更新時間 |

NBA PERSON_ID／PLAYER_ID／personId 在不同端點的命名不同，計畫統一為 player_id，但需用實際樣本確認對應。GAME_ID 存字串，保留前導零。若使用 BALLDONTLIE，明確保存 source 命名空間，不直接拼接兩家的整數 ID。

既有 `outputs/nba_2025_26/export_nba.mjs` 可見原始 rows 的 player／game 連結曾被解析；這只是程式碼觀察，**不是**原始檔完整或 ID 正確的驗證。後續先唯讀核對原始連結與 CSV 行對應，再另產生含 ID 的衍生檔。不得單靠姓名自動配對，也不改寫既有 CSV。跨供應商對照須人工確認姓名、球隊及其他識別資訊，歧義列隔離。

以 current_team_id 連接主客隊取得球員下一場；選該隊最早的合格未來例行賽。名單缺球隊、存在矛盾或無近期資料時顯示不可預測理由。此流程不判斷傷病或是否實際出賽，畫面持續標示「以球員會出賽為前提」。新球員的冷啟動規則另待定。

歷史資料需保留真正有出賽但零分的列；缺統計、DNP 或未出賽不補零。MIN 可能有不同精度／格式，不能僅依四捨五入後 `MIN=0` 判定 DNP；後續核對出賽資訊與來源語意。百分比統一 0–1，既有 CSV 的 FG% 尺度須核對。當場 MIN、FG_PCT、PTS 等只作結果或過去場特徵，不作當場賽前輸入。

## 小量唯讀連線結果

實測位置：本機 Windows PowerShell；未安裝套件、未登入、未申請帳號。每個請求 timeout 15 秒，不分頁、不掃描全季、不重試迴圈。普通沙箱三次請求遇到 DNS 解析失敗；按工具權限流程獲准後，原樣各重試一次，共六次 HTTP 嘗試。未繞過來源存取控制。

| 測試 | 沙箱外結果（2026-10-13 UTC，詳見 JSON） | 能支持的結論 |
|---|---|---|
| NBA CDN `scheduleLeagueV2.json` | HTTP 403，約 0.17 秒 | 已實測拒絕；無賽程 JSON，不能證明 2026-27 賽程已發布或不存在 |
| CommonAllPlayers，Season=2026-27，IsOnlyCurrentSeason=1 | 約 15 秒逾時，無 HTTP 狀態 | 已實測失敗；無法驗證目前球隊欄位的實際內容 |
| LeagueGameLog，PlayerOrTeam=P，Season=2025-26，僅 2026-04-12 | 約 15 秒逾時，無 HTTP 狀態 | 已實測失敗；未核對任何歷史列。此為窄日期歷史探測，不等同推薦 PlayerGameLogs 實測 |
| PlayerGameLogs、ScheduleLeagueV2 Stats、CommonTeamRoster | 未測 | 僅文件確認欄位與介面 |
| BALLDONTLIE stats／games／players | 未測 | 僅文件確認；需要 key，故未嘗試受保護端點 |

詳細記錄：[沙箱外結果](../../outputs/week1/source_probe/results.json)、[沙箱結果](../../outputs/week1/source_probe/results_sandbox.json)、[可重跑探測腳本](../../outputs/week1/source_probe/probe.ps1)。檔案包含 URL、參數、UTC 時間、耗時、錯誤及可得的 HTTP 狀態；本次沒有成功回應資料。瀏覽工具對 CDN 也未成功開啟，不能取代本機 HTTP 測試。

這些結果只反映該環境、時刻、請求參數與 headers；403／逾時原因未確定，不可推論所有網路或 Streamlit 一定被封鎖，也不可宣稱端點停用。

## 手動更新流程提案（尚未實作）

1. 操作者在本機明確執行一次更新，指定球季與時間範圍。2025-26 歷史與 2026-27 未來資料分開管理；不得因目前時間為十月就將季前賽混入例行賽。初次涵蓋範圍另確認；後續建議重取最近 7 個聯盟日期的完成比賽處理更正，名單與當季賽程重取。7 天是專案策略，不是來源保證。
2. 建立新的 staging 版本，記錄 attempt_at_utc。HTTP 成功後再檢查 JSON schema、球季／賽事類型、ID、時間、重複鍵、統計型別與隊伍關聯。分頁全部結束才算完整；單頁或部分成功不得發布。請求／格式／必要名單驗證失敗即終止，有限退避與重試次數需在正式實作定義，不無限重試。
3. 以 ID 合併歷史並保留更正紀錄，不寫回既有 CSV。名單缺球隊的合理個案可隔離並顯示不可預測；大規模名單縮減、球季不符、欄位消失、賽程異常清空則需要人工檢查，不能以「更新成功」掩蓋。
4. 所有必要表驗證通過後，寫入不可變 snapshot 與 manifest，最後一次切換 active manifest。保留上一成功版本；資料、manifest 必須同一版本，避免半套更新。若更新失敗，active 仍指舊版本，記錄錯誤與最後嘗試時間；成功時間保持不變。
5. 建議由操作者手動發布快照至選定持久儲存。若條款允許進 GitHub，可隨部署分支更新，Community Cloud 會同步儲存庫提交 [S2]；不在本次提交或推送。資料不能公開時，使用私有儲存／受控部署，方案待定。雲端程序本地檔案及快取不應作唯一持久來源（部署設計提案）。
6. Streamlit 讀 active snapshot，依 snapshot_id 更新讀取快取。介面分別顯示「資料成功更新時間」「歷史涵蓋至日期」「名單採集時間」「賽程採集時間」與「最近嘗試結果」，避免只用最新請求時間讓舊資料看似已更新。建議過 24 小時標示陳舊；這是專案提案。歷史回測仍可讀已凍結版本，未來預測需顯示時效與資料版本。

### 無未來賽程及異常處理提案

- 成功讀取正確球季且 schema 有效，但篩選後無合格賽程：顯示「目前沒有已確認的未來例行賽」，停用未來預測；歷史回測保留。休賽期、尚未發布、查詢窗口不足及球季切換各有可能，先檢查原因，不擅自補日期。
- 有比賽但無確定開賽時間、隊伍待定、延期、取消或狀態 unknown：不可預測；保留資料供查核。不要把零比分等同未開賽。NBA 的狀態碼與文字需用實際樣本建立對照；BALLDONTLIE 可依文件的 status_state 判定。
- 來源 403、逾時或資料解析錯誤：顯示「更新失敗，使用上次成功資料」及原成功時間。舊賽程仍需按當前時間排除已過截止場次；如所有舊賽程已過期，停用未來預測，不把網路失敗描述為官方無賽程。
- 空回應需分辨有效零筆、缺少預期結構與分頁失敗。合法空賽程可以發布空列表狀態；異常清空先保留舊版本。禁止用舊名單配新賽程而不揭露版本混用。

## 預測截止時間與日期／時區提案（未定案）

1. **每場開賽前 60 分鐘為截止時間**：`cutoff_utc = scheduled_start_utc - 60 minutes`。只允許 `now_utc < cutoff_utc`、狀態已確認未開賽、無延期／取消且團隊與時間完整的場次。截止後保留先前預測供查看，停止產生新預測；若無先前預測，顯示已過截止。
2. 選用截止前成功發布的最新一致快照；預測記錄 snapshot_id、generated_at_utc、cutoff_utc、schedule_observed_at。特徵只用在該時點前已完成且已取得的歷史比賽。手動更新不保證在截止前拿到當天最新資料，須明示滯後，不聲稱即時預測。跨場先後不可用只有日期的欄位假定資料已可得。
3. 儲存帶時區的 UTC 時間；UI 顯示台北 UTC+08:00，另顯示聯盟日期。NBA 賽程優先用 gameDateTimeUTC；不要把名為 Est 的欄位一律視為固定 UTC−5。若需東岸換日，使用 `America/New_York`，由時區庫處理夏令時間；台北使用 `Asia/Taipei`。
4. 保留來源 GAME_DATE／date 原值，不將台北換算日期覆寫歷史聯盟日期。同一天的回測資料不得跨訓練／驗證／測試集合，提案用經核對的聯盟日期統一切分。休息天數提案按東岸日曆日期差減一（下限零），歷史與未來一致；若改用實際小時需有歷史時間戳。
5. BALLDONTLIE 的 UTC datetime 與 date、NBA GAME_DATE 的日期語意需以跨日樣本核對，不直接假定三者相同；datetime 缺失時不補午夜。球季 ID 映射提案：NBA `2025-26` 對供應商起始年 `2025`，按實際回應驗證，不依日曆年份推導。
6. 延期／改時後使先前預測失效，保留舊預測與當時賽程供追溯，待新時間明確且新截止前再產生。歷史回測若沒有當時名單／發布快照，只能驗證過去比賽特徵的時間隔離，不能宣稱重現當時所有可得資訊或轉隊名單。

## 尚待確認與後續驗收

- 在允許的本機及實際 Streamlit 環境各做一次小樣本成功讀取；取得 Stats 賽程、名單與歷史欄位及狀態碼樣本。本次失敗不足以完成來源可用性驗收。
- 核對既有 CSV、原始連結 ID、出賽零分與 DNP、FG% 尺度、交易前後球隊；確認不因姓名配對製造錯列。歷史球季擴充與冷啟動規則仍待決定。
- 確認 2026-27 例行賽已發布範圍、NBA Cup 場次是否計入例行賽、延賽／待定時間及季前賽排除規則；不能只依 `postseason=false` 篩選。
- 若選 BALLDONTLIE，後續由使用者決定預算與帳號；驗證至少一場完整逐場資料、一筆跨隊球員與未來賽程，建立供應商 ID 對照。配額不是資料更新 SLA，兩者分開。
- 公開 GitHub 原始資料、快照及 Streamlit 功能的使用條件需要收斂；NBA 方案不能因免 key 或 nba_api MIT 就視為無限制。資料儲存方式與部署帳號尚待決定。
- 確認截止 60 分鐘、陳舊 24 小時、最近 7 天更正窗口、日期切分及休息天數規則；這些均是提案，沒有修改 PRD 的已確認決策。
- 正式更新驗收應涵蓋：來源故障、部分分頁、合法空賽程、錯誤球季、延期改時與 manifest 切換失敗；本次只列驗收要求，沒有實作服務或訓練模型。

## 查證來源

下列來源皆於 2026-10-13 查閱。文件存在／宣告與 HTTP 實測分開，不以文件中的歷史 success 欄位表示本次成功。

- [A1：nba_api PlayerGameLogs 文件](https://github.com/swar/nba_api/blob/master/docs/nba_api/stats/endpoints/playergamelogs.md)
- [A2：CommonAllPlayers 文件](https://github.com/swar/nba_api/blob/master/docs/nba_api/stats/endpoints/commonallplayers.md)；[目前原始碼欄位](https://github.com/swar/nba_api/blob/master/src/nba_api/stats/endpoints/commonallplayers.py)（有額外 slug，實作應按 headers 解析）。
- [A3：ScheduleLeagueV2 文件](https://github.com/swar/nba_api/blob/master/docs/nba_api/stats/endpoints/scheduleleaguev2.md)；[原始碼](https://github.com/swar/nba_api/blob/master/src/nba_api/stats/endpoints/scheduleleaguev2.py)；[NBA CDN 候選 URL](https://cdn.nba.com/static/json/staticData/scheduleLeagueV2.json)（本次 403，沒有成功驗證內容）。
- [A4：CommonTeamRoster 文件](https://github.com/swar/nba_api/blob/master/docs/nba_api/stats/endpoints/commonteamroster.md)
- [A5：nba_api 維護者 README／環境／授權](https://github.com/swar/nba_api)；[LeagueGameLog 探測端點文件](https://github.com/swar/nba_api/blob/master/docs/nba_api/stats/endpoints/leaguegamelog.md)（預設文件列團隊欄位，PlayerOrTeam=P 的實際 schema 未取得）。
- [A6：NBA 官方使用條款，第 9 節](https://www.nba.com/termsofuse)
- [A7：nba_api 靜態球員文件](https://github.com/swar/nba_api/blob/master/docs/nba_api/stats/static/players.md)
- [B1：BALLDONTLIE NBA 官方 API 文件](https://docs.balldontlie.io/)（供應商官方文件，非 NBA 官方來源）；[OpenAPI](https://www.balldontlie.io/openapi/nba.yml) 為該文件提供的 schema 入口，本次未另解析。
- [B2：BALLDONTLIE Account API／價格及配額文件](https://www.balldontlie.io/account/)
- [B3：BALLDONTLIE 使用條款，第 5–7 節](https://www.balldontlie.io/terms)
- [S1：Streamlit Community Cloud secrets 管理](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management)
- [S2：Streamlit 儲存庫更新及部署同步](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app/edit-your-app)

