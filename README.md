# 📈 台灣股市趨勢查詢系統

台灣股市股價查詢與 AI 預測 Web 應用，支援上市（.TW）及上櫃（.TWO）股票，提供互動式 K 線圖、成交量分析，以及基於 Random Forest 的未來兩周股價預測。

**線上 Demo：** https://stock-trend-lv4h.onrender.com/

---

## 功能特色

- **股票查詢**：輸入股票代碼，自動顯示「名稱(代碼)」格式，支援上市、上櫃
- **多時段切換**：1個月、3個月、1年、3年、5年
- **互動式 K 線圖**：開高低收蠟燭圖 + 成交量子圖，深色主題
- **即時行情卡片**：收盤價、漲跌幅、開盤、最高、最低、成交量
- **未來兩周預測**：Random Forest Regressor 遞迴預測未來 10 個交易日收盤價，附 95% 信賴區間

---

## 技術架構

| 層級 | 技術 |
|------|------|
| 後端 | Python / Flask |
| 資料來源 | yfinance（Yahoo Finance） |
| 圖表 | Plotly（互動式） |
| ML 模型 | scikit-learn RandomForestRegressor |
| 前端 | 原生 HTML / CSS / JavaScript |
| 部署 | Render（gunicorn） |

---

## ML 預測說明

### 特徵工程（11 個特徵）

| 特徵 | 說明 |
|------|------|
| MA5 / MA10 / MA20 | 5、10、20 日移動均線 |
| ret1 / ret5 | 1 日、5 日漲跌幅（Momentum） |
| Lag1 ~ Lag5 | 前 5 日滯後收盤價 |
| 短期波動率 | 近 5 日收盤價標準差 |

### 預測流程

1. 以近一年（約 250 個交易日）歷史資料訓練 Random Forest（300 棵樹）
2. 遞迴預測：將第 N 天預測值填回序列，計算第 N+1 天特徵，循環 10 次
3. 信賴區間：300 棵樹各自預測，取均值 ± 1.96σ 得 95% 預測區間

> ⚠️ 免責聲明：預測結果僅供參考，不構成任何投資建議。

---

## 本地執行

### 安裝依賴

```bash
pip install -r requirements.txt
```

### 啟動伺服器

```bash
python app.py
```

瀏覽器開啟 http://127.0.0.1:5000

---

## 專案結構

```
stock_trend/
├── app.py              # Flask 主程式（路由、資料處理、ML 預測）
├── templates/
│   └── index.html      # 前端單頁應用
├── requirements.txt    # Python 依賴套件
├── render.yaml         # Render 部署設定
└── deploy_web.txt      # 線上部署網址
```

---

## API 端點

| 端點 | 方法 | 參數 | 說明 |
|------|------|------|------|
| `/` | GET | — | 主頁面 |
| `/stock` | GET | `id`, `period` | 查詢股票資料與圖表 |
| `/predict` | GET | `id` | 預測未來兩周股價 |

**period 可選值：** `1m` / `3m` / `1y` / `3y` / `5y`

---

## 部署（Render）

專案已包含 `render.yaml`，連結 GitHub repo 後 Render 會自動讀取設定部署：

```yaml
services:
  - type: web
    runtime: python
    buildCommand: pip install -r requirements.txt
    startCommand: gunicorn app:app --bind 0.0.0.0:$PORT
```

---

## 資料來源

- 股價資料：[Yahoo Finance](https://finance.yahoo.com/) via yfinance
