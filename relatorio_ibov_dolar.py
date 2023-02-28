import pandas as pd
import datetime
import yfinance as yf
from matplotlib import pyplot as plt
import mplcyberpunk

tickers_de_negociacao = ['^BVSP','BRL=X']

hoje = datetime.datetime.now()

um_ano_atras = hoje - datetime.timedelta(days = 365)

dados_mercado = yf.download(tickers_de_negociacao, um_ano_atras, hoje)

dados_fechamento = dados_mercado['Adj Close']

dados_fechamento.columns = ['dolar', 'ibovespa']

dados_fechamento = dados_fechamento.dropna()

dados_anuais = dados_fechamento.resample('Y').last()

dados_mensais = dados_fechamento.resample('M').last()

retorno_anual = dados_anuais.pct_change().dropna()

retorno_mensal = dados_mensais.pct_change().dropna()

retorno_diario = dados_fechamento.pct_change().dropna()

retorno_diario_dolar = retorno_diario.iloc[-1,0]
retorno_diario_ibov = retorno_diario.iloc[-1,1]

retorno_mensal_dolar = retorno_mensal.iloc[-1,0]
retorno_mensal_ibov = retorno_mensal.iloc[-1,1]

retorno_anual_dolar = retorno_anual.iloc[-1,0]
retorno_anual_ibov = retorno_anual.iloc[-1,1]

retorno_diario_dolar = round((retorno_diario_dolar * 100), 2)
retorno_diario_ibov = round((retorno_diario_ibov * 100), 2)
print("Daily dollar return:", retorno_diario_dolar, "%")
print("Daily Ibovespa return:", retorno_diario_ibov, "%")

retorno_mensal_dolar = round((retorno_mensal_dolar * 100), 2)
retorno_mensal_ibov = round((retorno_mensal_ibov * 100), 2)
print("Monthly dollar return:", retorno_mensal_dolar, "%")
print("Monthly Ibovespa return:", retorno_mensal_ibov, "%")

retorno_anual_dolar = round((retorno_anual_dolar * 100), 2)
retorno_anual_ibov = round((retorno_anual_ibov * 100), 2)
print("Annual dollar return:", retorno_anual_dolar, "%")
print("Annual Ibovespa return:", retorno_anual_ibov, "%")


plt.style.use('cyberpunk')

dados_fechamento.plot(y = 'ibovespa', use_index = True, legend = False)
plt.title('ibovespa')
plt.show()

plt.style.use('cyberpunk')

dados_fechamento.plot(y = 'dolar', use_index = True, legend = False)
plt.title('dolar')
plt.show()
