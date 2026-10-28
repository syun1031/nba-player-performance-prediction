# 專案 Python 環境

專案使用根目錄的 `.venv`，與其他 Python 環境分開。既有 `.venv-nba` 保留，本次不修改。

在專案根目錄的 PowerShell 可直接執行：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

不需先啟用環境；直接指定這個 Python 路徑即可避免用錯環境。

已安裝 Scikit-learn、XGBoost、Matplotlib 及其相依套件。完整固定版本見根目錄 `requirements-modeling.txt`。此檔只記錄本次建模環境，尚非含 Streamlit 的完整應用需求。

其他電腦以 Python 3.12 建立環境後，可使用：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-modeling.txt
```

`.venv` 不應上傳 GitHub；提交版本清單，讓其他人自行建立環境。
