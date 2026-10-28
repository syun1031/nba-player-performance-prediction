# NBA 球員得分研究室

用歷史資料預測球員單場得分，並透過 Streamlit 查看每場預測與模型誤差。已完成資料處理、模型訓練、最終測試，以及歷史回測／模型比較兩頁；尚未提供未來比賽預測或雲端部署。

## 開啟介面

本機服務運作時，直接開啟：

- 歷史回測：http://127.0.0.1:8501/
- 模型比較：http://127.0.0.1:8501/comparison

若要重新啟動，在專案根目錄的 PowerShell 執行：

```powershell
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py --server.address 127.0.0.1 --server.port 8501
```

停止服務可在執行它的終端按 Ctrl+C。目前只驗證本機 Windows／Python 3.12；GitHub 私人儲存庫已建立，尚未完成 Streamlit Community Cloud 部署。

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

再依上方步驟建立環境與啟動介面。`.gitignore` 排除虛擬環境、`node_modules/`、Python 暫存及本機認證設定；這些檔案不用上傳。舊企劃 Word、根目錄一次性文件修改腳本與空原始 JSON 保留在本機，不納入初始提交。雲端部署另行處理。
