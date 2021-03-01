import yfinance as yf

tsla = yf.Ticker("TSLA")
tsla_stock = tsla.history(
    start="2020-10-01", end="2020-12-30"
).reset_index()