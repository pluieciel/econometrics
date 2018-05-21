
gdp=France.GDP$Value
plot(gdp,type="l")
ts(gdp)
plot(ts(gdp,start=1961,frequency=4))
gdpacf=acf(gdp)
Box.test(gdp,lag=5,type="Ljung-Box",fitdf = 0)

T=length(gdp)

ar1model=lm(gdp[2:T]~gdp[1:T-1])
summary(ar1model)
Box.test(ar1model$residuals,lag=5,type="Ljung-Box",fitdf = 0)

mforecast=AR1_forecast(gdp,0.2,4)

forecast_performance(gdp,0.2,mforecast)