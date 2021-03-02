from fbprophet import Prophet

# Python
m = Prophet()
m.fit(df)
# Python
future = m.make_future_dataframe(periods=30)
future.tail()
# Python
forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
# Python
fig1 = m.plot(forecast)