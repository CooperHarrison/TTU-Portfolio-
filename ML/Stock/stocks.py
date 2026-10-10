# Stock movement experiment:
# - Downloads recent daily OHLCV data for up to 40 S&P 500 stocks, builds rolling
#   features, trains an LSTM, and predicts the next-day movement for the latest
#   available date.
# - Writes the 20 highest predicted movers and their estimated prices to
#   top20_next_day_with_prices.csv.
# - Run with: py "ML/Stock/stocks.py"
# - Requires internet access and: py -m pip install numpy pandas requests yfinance torch lxml
# - This is an experimental model, not financial advice.

import time
import logging
from io import StringIO
import random

import requests
import numpy as np
import pandas as pd
import yfinance as yf
import torch
import torch.nn as nn


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

SEQ_LEN = 30
MIN_HISTORY = 60
UNIVERSE_SIZE = 40
TRAIN_YEARS = 3
PERIOD = f"{TRAIN_YEARS}y"
INTERVAL = "1d"

DEVICE = torch.device("cpu")

PERIODS_TO_TRY = [PERIOD, "1y", "6mo", "3mo"]

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

HARD_CODED_SAFE_TICKERS = [
    "AAPL","MSFT","AMZN","GOOGL","META","TSLA","NVDA","BRK-B","JPM","JNJ",
    "V","PG","UNH","HD","MA","DIS","BAC","XOM","CVX","KO",
    "PFE","MRK","ABBV","ORCL","ADBE","CMCSA","NFLX","INTC","T","PEP",
    "CSCO","ABT","CRM","NKE","MCD","WMT","LLY","TXN","QCOM","BMY",
    "COST","SBUX","AMGN","MDT","NEE","HON","UPS","RTX","LOW","AXP"
]


def get_sp500_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        tables = pd.read_html(StringIO(r.text))
        return tables[0]
    except Exception:
        return pd.DataFrame(columns=["Symbol"])

def to_yf_symbol(t):
    return t.replace(".", "-")

def has_required_columns(df):
    required = ["Open", "High", "Low", "Close", "Volume"]
    return all(col in df.columns for col in required)

def normalize_hist(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

   
    close_col = None
    for c in df.columns:
        if "close" in c.lower():
            close_col = c
            break
    if close_col is None:
        return None

    df["Close"] = pd.to_numeric(df[close_col], errors="coerce")

    for name in ["Open","High","Low","Volume"]:
        found = None
        for c in df.columns:
            if name.lower() in c.lower():
                found = c
                break
        if found is None:
            return None
        df[name] = pd.to_numeric(df[found], errors="coerce")

    df = df[["Open","High","Low","Close","Volume"]].dropna()
    return df if len(df) > 0 else None

def download_one(t):
    for p in PERIODS_TO_TRY:
        try:
            df = yf.download(t, period=p, interval=INTERVAL, progress=False)
            if df is not None and len(df) > 0:
                df = normalize_hist(df)
                if df is not None and has_required_columns(df) and len(df) >= MIN_HISTORY:
                    return df
        except Exception:
            pass
    return None

def download_universe():
    df = get_sp500_tickers()
    tickers = df["Symbol"].tolist() if not df.empty else HARD_CODED_SAFE_TICKERS

    data = {}
    for t in tickers:
        yf_t = to_yf_symbol(t)
        hist = download_one(yf_t)
        if hist is not None:
            data[t] = hist
        if len(data) >= UNIVERSE_SIZE:
            break

    return data


def build_features(df):
    if not has_required_columns(df):
        return None

    df = df.copy().sort_index()
    df["intraday_ret"] = (df["Close"] - df["Open"]) / df["Open"]
    df["ret1"] = df["Close"].pct_change()
    df["ma5"] = df["Close"].rolling(5).mean()
    df["ma10"] = df["Close"].rolling(10).mean()
    df["mom5"] = df["Close"].pct_change(5)
    df["vol10"] = df["ret1"].rolling(10).std()
    df["vol20"] = df["ret1"].rolling(20).std()
    df["volume_avg10"] = df["Volume"].rolling(10).mean()
    df = df.dropna()
    return df if len(df) > 0 else None

def make_sequences(universe):
    X, y, meta = [], [], []
    for t, df in universe.items():
        f = build_features(df)
        if f is None or len(f) < SEQ_LEN + 1:
            continue

        for i in range(SEQ_LEN - 1, len(f) - 1):
            window = f.iloc[i-(SEQ_LEN-1):i+1]
            target = f["intraday_ret"].iloc[i+1]

            closes = window["Close"].values
            ret_seq = np.concatenate([[0.0], np.diff(closes)/closes[:-1]])

            feat = np.vstack([
                ret_seq,
                window["ma5"].values / closes,
                window["ma10"].values / closes,
                window["mom5"].values,
                window["vol10"].values,
                window["vol20"].values,
                window["volume_avg10"].values / (window["volume_avg10"].mean() + 1e-9)
            ]).T.astype(np.float32)

            X.append(feat)
            y.append(np.float32(target))
            meta.append((t, f.index[i]))

    return np.stack(X), np.array(y).reshape(-1,1), meta


class LSTMModel(nn.Module):
    def __init__(self, n_features, hidden=64):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

def train_model(X, y, n_features):
    model = LSTMModel(n_features).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    ds = torch.utils.data.TensorDataset(
        torch.from_numpy(X).float(),
        torch.from_numpy(y).float()
    )
    dl = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=True)

    for epoch in range(8):
        total = 0
        for xb, yb in dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            total += loss.item() * xb.size(0)
        logger.info(f"Epoch {epoch+1}/8 MSE {total/len(ds):.6f}")

    return model


def top20_predictions(universe, model, meta_map, X, date):
    entries = [(t, idx) for (t, d), idx in meta_map.items() if d == date]
    tickers = [t for t, _ in entries]

    Xb = torch.from_numpy(np.stack([X[idx] for _, idx in entries])).float().to(DEVICE)

    with torch.no_grad():
        preds = model(Xb).cpu().numpy().reshape(-1)

    results = []
    for i, t in enumerate(tickers):
        pred = float(preds[i])
        df = universe[t]

        pos = df.index.get_loc(date)
        prev_close = float(df["Close"].iloc[pos])

        predicted_close = prev_close * (1 + pred)
        delta = predicted_close - prev_close

        results.append({
            "ticker": t,
            "date": date,
            "previous_close": prev_close,
            "predicted_close": predicted_close,
            "delta": delta,
            "pred": pred
        })

    return sorted(results, key=lambda x: x["pred"], reverse=True)[:20]


def main():
    logger.info("Downloading universe...")
    universe = download_universe()

    logger.info("Building sequences...")
    X, y, meta = make_sequences(universe)
    X = X.astype(np.float32)
    y = y.astype(np.float32)

    n_features = X.shape[2]

    logger.info("Training model...")
    model = train_model(X, y, n_features)

    meta_map = {(t, d): i for i, (t, d) in enumerate(meta)}
    last_date = max(d for _, d in meta_map)

    logger.info(f"Predicting top 20 for {last_date}...")
    top20 = top20_predictions(universe, model, meta_map, X, last_date)

    df = pd.DataFrame(top20)
    df.to_csv("top20_next_day_with_prices.csv", index=False)
    print(df)

if __name__ == "__main__":
    main()
  