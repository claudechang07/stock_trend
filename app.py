from flask import Flask, render_template, request, jsonify
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

app = Flask(__name__)


PERIOD_DAYS = {
    "1m": 30,
    "3m": 90,
    "1y": 365,
    "3y": 365 * 3,
    "5y": 365 * 5,
}

PERIOD_LABELS = {
    "1m": "近一個月",
    "3m": "近三個月",
    "1y": "近一年",
    "3y": "近三年",
    "5y": "近五年",
}


def get_stock_data(stock_id, period="1y"):
    """Fetch stock data for Taiwan stock market for the given period."""
    end = datetime.today()
    days = PERIOD_DAYS.get(period, 365)
    start = end - timedelta(days=days)

    for suffix in [".TW", ".TWO"]:
        ticker_symbol = stock_id + suffix
        ticker = yf.Ticker(ticker_symbol)
        df = ticker.history(start=start, end=end)
        if not df.empty:
            try:
                info = ticker.info
                name = info.get("shortName") or info.get("longName") or stock_id
            except Exception:
                name = stock_id
            return df, ticker_symbol, name

    return None, None, None


def build_chart(df, stock_id, ticker_symbol, period="1y", name=""):
    """Build Plotly chart with price and volume subplots."""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("股價 (收盤價)", "成交量"),
        row_heights=[0.7, 0.3],
    )

    dates  = df.index.strftime("%Y-%m-%d").tolist()
    open_  = df["Open"].tolist()
    high   = df["High"].tolist()
    low    = df["Low"].tolist()
    close  = df["Close"].tolist()
    volume = df["Volume"].tolist()

    fig.add_trace(
        go.Candlestick(
            x=dates, open=open_, high=high, low=low, close=close,
            name="股價",
            increasing_line_color="#e8534a",
            decreasing_line_color="#4caf50",
        ),
        row=1, col=1,
    )

    colors = ["#e8534a" if c >= o else "#4caf50" for c, o in zip(close, open_)]
    fig.add_trace(
        go.Bar(x=dates, y=volume, name="成交量", marker_color=colors, opacity=0.8),
        row=2, col=1,
    )

    fig.update_layout(
        title=dict(
            text=f"{name}({stock_id}) {PERIOD_LABELS.get(period, '近一年')}股價與成交量趨勢",
            font=dict(size=18),
        ),
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        paper_bgcolor="#1e1e2e",
        plot_bgcolor="#1e1e2e",
        font=dict(color="#cdd6f4"),
        height=650,
        margin=dict(l=40, r=40, t=80, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#313244")
    fig.update_yaxes(showgrid=True, gridcolor="#313244")

    return json.loads(pio.to_json(fig, engine="json"))


# ── ML prediction ────────────────────────────────────────────────────────────

def _make_features(closes: np.ndarray) -> np.ndarray:
    """Build feature vector from a window of closing prices (length >= 20)."""
    ma5  = np.mean(closes[-5:])
    ma10 = np.mean(closes[-10:])
    ma20 = np.mean(closes[-20:])
    ret1 = (closes[-1] - closes[-2]) / closes[-2]
    ret5 = (closes[-1] - closes[-6]) / closes[-6] if len(closes) >= 6 else 0.0
    lags = closes[-5:][::-1].tolist()          # lag1 … lag5 (newest first)
    high_low_vol = np.std(closes[-5:])         # short-term volatility proxy
    return np.array([ma5, ma10, ma20, ret1, ret5, *lags, high_low_vol])


def predict_next_week(df: pd.DataFrame):
    """
    Train a Random Forest Regressor on engineered lag/MA features derived
    from 1-year of daily closing prices, then recursively predict the next
    10 trading days.  Returns (dates, preds, lower_95, upper_95).
    """
    from sklearn.ensemble import RandomForestRegressor

    closes = df["Close"].values.astype(float)

    if len(closes) < 30:
        raise ValueError("歷史資料不足（需至少 30 個交易日）")

    # Build training set: feature window → next-day close
    X, y = [], []
    for i in range(20, len(closes) - 1):
        X.append(_make_features(closes[: i + 1]))
        y.append(closes[i + 1])
    X, y = np.array(X), np.array(y)

    model = RandomForestRegressor(
        n_estimators=300,
        max_features="sqrt",
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X, y)

    # Recursive 10-day forecast
    history = list(closes)
    pred_vals, lower_vals, upper_vals = [], [], []

    for _ in range(10):
        feats = _make_features(np.array(history)).reshape(1, -1)
        # Collect each tree's prediction for a 95 % confidence interval
        tree_preds = np.array([t.predict(feats)[0] for t in model.estimators_])
        p   = float(np.mean(tree_preds))
        std = float(np.std(tree_preds))
        pred_vals.append(round(p, 2))
        lower_vals.append(round(p - 1.96 * std, 2))
        upper_vals.append(round(p + 1.96 * std, 2))
        history.append(p)

    # Generate next 10 trading-day dates (skip weekends)
    last_date = df.index[-1].to_pydatetime().date()
    future_dates = []
    d = last_date
    while len(future_dates) < 10:
        d += timedelta(days=1)
        if d.weekday() < 5:            # Mon–Fri
            future_dates.append(d.strftime("%Y-%m-%d"))

    return future_dates, pred_vals, lower_vals, upper_vals


def build_prediction_chart(df, stock_id, ticker_symbol,
                            future_dates, preds, lower, upper, name=""):
    """Show last 60 trading days + 5-day forecast on a single price chart."""
    tail = df.tail(60)
    hist_dates  = tail.index.strftime("%Y-%m-%d").tolist()
    hist_close  = tail["Close"].tolist()
    hist_volume = tail["Volume"].tolist()
    hist_open   = tail["Open"].tolist()

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("股價：近60日 + 未來10個交易日預測", "成交量（近60日）"),
        row_heights=[0.72, 0.28],
    )

    # Historical line
    fig.add_trace(
        go.Scatter(
            x=hist_dates, y=hist_close,
            mode="lines", name="歷史收盤價",
            line=dict(color="#89b4fa", width=2),
        ),
        row=1, col=1,
    )

    # Confidence band (filled area)
    band_x = [future_dates[0]] + future_dates + [future_dates[-1]] + future_dates[::-1]
    band_y = [lower[0]] + upper + [upper[-1]] + lower[::-1]
    fig.add_trace(
        go.Scatter(
            x=band_x, y=band_y,
            fill="toself",
            fillcolor="rgba(249,226,175,0.18)",
            line=dict(color="rgba(0,0,0,0)"),
            name="95% 預測區間",
            hoverinfo="skip",
        ),
        row=1, col=1,
    )

    # Prediction line
    connect_x = [hist_dates[-1]] + future_dates
    connect_y = [hist_close[-1]] + preds
    fig.add_trace(
        go.Scatter(
            x=connect_x, y=connect_y,
            mode="lines+markers",
            name="預測收盤價",
            line=dict(color="#f9e2af", width=2.5, dash="dash"),
            marker=dict(size=7, color="#f9e2af"),
        ),
        row=1, col=1,
    )

    # Volume bars
    vol_colors = [
        "#e8534a" if c >= o else "#4caf50"
        for c, o in zip(hist_close, hist_open)
    ]
    fig.add_trace(
        go.Bar(x=hist_dates, y=hist_volume, name="成交量",
               marker_color=vol_colors, opacity=0.7),
        row=2, col=1,
    )

    fig.update_layout(
        title=dict(
            text=f"{name}({stock_id}) 未來兩周股價預測（Random Forest）",
            font=dict(size=18),
        ),
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        paper_bgcolor="#1e1e2e",
        plot_bgcolor="#1e1e2e",
        font=dict(color="#cdd6f4"),
        height=650,
        margin=dict(l=40, r=40, t=80, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#313244")
    fig.update_yaxes(showgrid=True, gridcolor="#313244")

    return json.loads(pio.to_json(fig, engine="json"))


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/stock", methods=["GET"])
def stock():
    stock_id = request.args.get("id", "").strip()
    period   = request.args.get("period", "1y")
    if period not in PERIOD_DAYS:
        period = "1y"
    if not stock_id:
        return jsonify({"error": "請輸入股票編號"}), 400

    df, ticker_symbol, name = get_stock_data(stock_id, period)
    if df is None or df.empty:
        return jsonify({"error": f"找不到股票「{stock_id}」的資料，請確認股票編號是否正確"}), 404

    chart_data = build_chart(df, stock_id, ticker_symbol, period, name)

    latest = df.iloc[-1]
    prev   = df.iloc[-2] if len(df) > 1 else latest
    change = latest["Close"] - prev["Close"]
    change_pct = (change / prev["Close"]) * 100

    info = {
        "close":      round(latest["Close"], 2),
        "open":       round(latest["Open"],  2),
        "high":       round(latest["High"],  2),
        "low":        round(latest["Low"],   2),
        "volume":     int(latest["Volume"]),
        "change":     round(change, 2),
        "change_pct": round(change_pct, 2),
        "date":       df.index[-1].strftime("%Y-%m-%d"),
    }

    return jsonify({"chart": chart_data, "info": info, "ticker": ticker_symbol, "name": name})


@app.route("/predict", methods=["GET"])
def predict():
    stock_id = request.args.get("id", "").strip()
    if not stock_id:
        return jsonify({"error": "請輸入股票編號"}), 400

    # Always fetch 1 year of data for model training
    df, ticker_symbol, name = get_stock_data(stock_id, "1y")
    if df is None or df.empty:
        return jsonify({"error": f"找不到股票「{stock_id}」的資料"}), 404

    try:
        future_dates, preds, lower, upper = predict_next_week(df)
    except Exception as e:
        return jsonify({"error": f"預測失敗：{str(e)}"}), 500

    chart_data = build_prediction_chart(
        df, stock_id, ticker_symbol, future_dates, preds, lower, upper, name
    )

    table = [
        {"date": d, "price": p, "lower": l, "upper": u}
        for d, p, l, u in zip(future_dates, preds, lower, upper)
    ]

    return jsonify({"chart": chart_data, "table": table, "ticker": ticker_symbol})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
