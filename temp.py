import yfinance as yf

ticker = "SPY"
start_date = "2023-01-01"
end_date = "2024-01-01"

data = yf.download(ticker, start=start_date, end=end_date)
print(data)
