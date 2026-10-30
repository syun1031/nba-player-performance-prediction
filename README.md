# NBA 球員得分研究室

**線上模型比較已同步兩季研究結果。**開啟 [模型比較頁](https://nba-player-performance-prediction1.streamlit.app/comparison)，在「比較範圍」選「三個研究候選」，即可切換2025–26／2024–25、CV／二月／三月至季末與高分分組。GitHub與線上驗收詳見PROGRESS.md。

模型比較頁新增 **2025–26／2024–25球季**與 **三個研究候選**：原九項、原九項＋上季表現、1＋3＋同區＋上季表現＋2倍。兩季均可查看CV／二月／三月至季末與20+/30+分組，下載逐場預測。2025–26三月新增結果見 [追加比較報告](outputs/season_test_2025_26_prior_models/report.md)，屬已使用測試上的研究比較，並非新的獨立最終測試。

**最新：2024–25三月至季末三模型已完成一次性預測。**[測試報告](outputs/season_test_2024_25_three_models/report.md)、[三模型逐場CSV](outputs/season_test_2024_25_three_models/三模型逐場預測_2025年3月至季末.csv)。7060筆共同樣本，原九項／原九項上季／組合上季2倍的整體MAE為4.7738／4.7434／4.7620，30+為13.0315／12.9784／11.5995。本輪0fit、189測試通過，線上不變。最新test_usage已消耗一次；較早封存檔案是歷史快照。唯讀核對：`.venv\Scripts\python.exe -X utf8 -m scripts.verify_march_2024`。

目前保留三候選：只有原九項、原九項＋上季表現、1＋3＋同區＋上季表現＋2倍。[2024–25三模型結果](outputs/season_validation_2024_25_three_models/report.md)及[二月逐場預測](outputs/season_validation_2024_25_three_models/三模型逐場預測_February.csv)已完成。二月整體MAE依序4.7055／4.6531／4.6934，30+為12.8412／12.5698／11.0156。三月後已依後續指示完成一次性測試，線上模型不變。2023–24 [正分鐘建模資料](outputs/nba_2023_24/positive_minute_model_data.csv)及[來源說明](outputs/nba_2023_24/README.md)；2024–25原CSV不變，補充資料在 outputs/nba_2024_25/supplement_2026_10_30/。三模型唯讀驗證：`.venv\Scripts\python.exe -X utf8 -m scripts.verify_three_models_2024`。

上季表現研究：[v9報告](outputs/week2/prior_season_profile_v9/report.md)。已比較「原九項＋上季表現」與「1＋3＋同區＋上季表現＋2倍」。後者二月MAE由4.6326降至4.6246、30+由10.8492降至10.4281；整體改善仍小，無上季資料者與30分警示並未同步變好，尚未換線上模型。唯讀核對：`.venv\Scripts\python.exe -X utf8 -m scripts.verify_prior_season_profile`。

相對五場均值的增減研究已完成：[v8報告](outputs/week2/residual_offset_v8/report.md)、[Luka逐場CSV](outputs/week2/residual_offset_v8/luka_demo/Luka_均值增減預測.csv)。Luka同17場MAE從舊加權模型9.20降到7.99，但五場均值仍較佳（7.49）；全體CV／二月未勝過對應直接模型，因此保留為研究版本。唯讀驗證可執行 `.venv\Scripts\python.exe -X utf8 -m scripts.verify_residual_offset`，不重新訓練。

用歷史資料預測球員單場得分，並透過 Streamlit 查看每場預測與模型誤差。已完成資料處理、模型訓練、最終測試，以及歷史回測／模型比較兩頁；未來比賽預測尚未提供。線上服務：[NBA得分研究室](https://nba-player-performance-prediction1.streamlit.app/)。

## 開啟介面

本機服務運作時，直接開啟：

- 歷史回測：http://127.0.0.1:8501/
- 模型比較：http://127.0.0.1:8501/comparison

若要重新啟動，在專案根目錄的 PowerShell 執行：

```powershell
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py --server.address 127.0.0.1 --server.port 8501
```

停止本機服務可在執行它的終端按 Ctrl+C。GitHub私人儲存庫已連接Streamlit Community Cloud；線上更新的實際驗收紀錄見PROGRESS.md。

## 兩頁怎麼用

**歷史回測**：選球員，再選測試期的一場比賽，查看模型預測、實際得分、絕對誤差與五場均值。近期十場只含該場之前的正分鐘出賽；下方另有清楚標示的完整測試期曲線與紀錄，可下載 CSV。

**模型比較**：切換二月驗證或最終測試，查看三種模型、不同特徵版本及五場均值基準。可篩選實際20分／30分以上場次，每種模型都用同一批樣本。篩選只作展示，不重新選模；頁面卡片始終顯示鎖定主模型。

MAE代表平均差幾分；RMSE較重視大誤差，兩者越低越好。殘差定義為實際減預測，正值為低估、負值為高估。

## 已鎖定模型與結果

主模型為 XGBoost，使用九項特徵：近五場／最多十場平均得分、近五場平均分鐘／籃板／助攻、主場、中立場、個人休息天數與自己是否先發。模型及前處理只擬合到2026-01-31，二月選模，三月一日至四月十二日作最終測試。

| 最終測試組 | 筆數 | 主模型 MAE | 五場均值 MAE |
| --- | ---: | ---: | ---: |
| 全部合格出賽 | 6,979 | 4.7475 | 4.9444 |
| 實際20分以上 | 1,103 | 8.5597 | 8.3927 |
| 實際30分以上 | 228 | 13.7645 | 12.7456 |

主模型整體 RMSE為6.2132，MAE比基準低約3.98%；高得分場次仍較差。20+與30+互相重疊，不能相加。完整精度與說明以 [最終測試報告](outputs/week2/final_test/README.md) 為準。

## 資料與限制

另完成五項近期變化／波動特徵的十四項研究版本，固定原參數比較；二月MAE只從4.6115降至4.6050，30+略退步，目前不採用。研究模型與結果保存在 [研究報告](outputs/week2/recent_change_v2/report.md)，不影響上述正式模型與介面。三月至四月未重新評分；後續需新資料驗證。

另有 [預估分鐘×得分效率兩階段研究](outputs/week2/two_stage_v3/report.md)：二月得分MAE 4.5977，三輪合併4.5719，改善不穩定且30+較差，因此仍保留原正式模型。分鐘預估本身較五場均值改善約5.90%～7.27%，留作後續研究；原測試未重新評分。

第三項 [對手背景研究](outputs/week2/opponent_context_v4/report.md) 新增對手近五場失分與雙方總分平均。共同訓練十四項→十六項的二月MAE 4.6027→4.5956、CV 4.5661→4.5634，改善很小，二月30+反而較差，不替換正式模型。完整原始隊伍加總與名單已核對；總分是得分環境代理，不是真正Pace，仍需補回合數來源。

[四種組合比較](outputs/week2/optimization_combinations_v5/report.md) 已用相同共同樣本完成28次固定fit：CV以1＋3最佳（MAE4.5634），二月以1＋2＋3最佳（4.5909）；兩期冠軍不同且差異很小。1＋3在兩期高分群組與二月RMSE較好，優先保留為研究候選，仍不更換原正式模型。完整四組排名、12個成對差異與限制見報告。

[同區對陣記憶研究](outputs/week2/same_conference_h2h_v6/report.md) 新增是否同區、此前交手次數、最近／平均得分。沒有紀錄留空不補0，二月同區紀錄覆蓋率78.6%。加到1＋3後，CV MAE4.5634→4.5621、二月4.5956→4.5926，改善很小；高分略好，仍維持原正式模型。只支持已核對的2025–26分區版本，未使用舊測試重選模型。

[高分樣本加權研究](outputs/week2/high_score_weighting_v7/report.md) 已先測九項，再加入1＋3＋同區20項，各用30+權重2／3倍共16fit。20項2倍的二月30+ MAE12.3160→10.8492，整體4.5926→4.6326；高分改善但一般場次與誤報變多，AP沒有一致改善。157項測試通過，僅保留研究候選，線上原模型不變。

依使用者指定另作 [Luka三月起展示](outputs/demos/luka_combined_w2_march_2026/report.md)：固定上述加權模型對3/1～4/2全部17場推論，MAE9.20，原模型9.82、五場均值7.49；CSV與曲線同目錄。這是已使用資料的探索展示，不是新測試，沒有重訓或換線上模型。

- 這是日期隔離的假設性回測，以球員實際出賽為前提。歷史先發來自賽後快照，不能稱為完整賽前資訊重建。
- 個人歷史只用目標日期前一天或更早資料；個人與球隊都需至少五場歷史。零分鐘不納入個人樣本或近期窗口，正分鐘零分保留。
- 資料期間：2025–26例行賽。畫面「回測成果版本」來自原成果日期，不是今天重新爬取資料的時間。
- 頁面只讀既有預測與原始歷史，不載入模型權重、不呼叫 fit 或 predict，也不重跑測試。篩選後指標只是既有預測的描述性彙整。
- 回測資料已有最終評分，不應再拿它調參或重新選模後宣稱是未看過的測試。未來改善需另保留資料。
- 未來模式仍需新球季來源、賽程、正式先發與手動更新驗收。尚未提供先發隊友缺席分析。
- 檔案缺漏或版本不一致時，頁面會停止展示並提示；不補造預測。

## 安裝與測試

已有 `.venv` 的本機不需重裝。新環境使用 Python 3.12：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

`requirements.txt` 保存完整應用版本；原建模版本另外保留在 `requirements-modeling.txt`。新增 Streamlit 1.66.1、Pandas 3.1.0、Altair 6.3.0，原 NumPy／Scikit-learn／XGBoost版本未更換。AppTest在部分受限制環境需要可用的暫存目錄權限；一般本機終端可直接執行。

## 檔案位置

- `streamlit_app.py`、`dashboard_pages/`：入口與兩頁導航。
- `nba_dashboard/data.py`：資料版本核對、同樣本指標、嚴格前日近期紀錄。
- `nba_dashboard/views.py`：繁體中文介面、圖表、篩選與下載。
- `outputs/week2/final_test/`：主模型、測試協議、既有預測及最終結果。
- `outputs/week2/time_validation_tuning/`：三輪驗證、固定共用參數與二月預測。
- `outputs/week3/streamlit_qa/`：操作畫面與本機驗收紀錄。
- [TODO](TODO.md)、[PROGRESS](PROGRESS.md)、[PRD](PRD.md)：範圍、進度與待辦。

請保留既有 `outputs/` 與 `outputs/nba_2025_26/` 的相對位置。介面會驗證使用到的凍結成果、原始歷史及其證據檔；只複製入口程式無法展示資料。`.venv` 不需放進未來的 Git 儲存庫，其他環境由版本清單重新建立。

介面採用 Streamlit 官方的 [多頁導航](https://docs.streamlit.io/develop/concepts/multipage-apps/page-and-navigation) 與 [AppTest](https://docs.streamlit.io/develop/api-reference/app-testing/st.testing.v1.apptest) 驗證方式。

## Git 儲存與成果完整性

Git 保留程式、資料、凍結模型、預測與驗證證據的相對位置；`.gitattributes` 禁止自動轉換換行，避免不同系統下載後 SHA-256 不一致。請勿批次重新格式化 `outputs/` 中的 CSV、JSON 或其他凍結成果。

私人儲存庫：[nba-player-performance-prediction](https://github.com/syun1031/nba-player-performance-prediction)。需要儲存庫存取權限才能下載。登入 GitHub CLI 後可執行：

```powershell
gh repo clone syun1031/nba-player-performance-prediction
cd nba-player-performance-prediction
```

再依上方步驟建立環境與啟動介面。`.gitignore` 排除虛擬環境、`node_modules/`、Python暫存及本機認證設定；這些檔案不用上傳。舊企劃Word、根目錄一次性文件修改腳本與空原始JSON保留在本機。線上app由部署main分支更新。
