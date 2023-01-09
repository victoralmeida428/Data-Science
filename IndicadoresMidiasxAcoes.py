import seaborn as sns
from pandas_datareader import data as pdr
import datetime as dt
import yfinance as yf
from pytrends.request import TrendReq
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr
import seaborn as sns

yf.pdr_override()

startdate = dt.datetime(2022,5,7)
enddate = dt.datetime(2023,1,7)
y_symbols = ['PETR4.SA']
pesquisa = ['petrobras']
data = pdr.get_data_yahoo(y_symbols, startdate, enddate)

trends = TrendReq(hl='pt-BR', tz=300, retries=2, backoff_factor=0.1)

trends.build_payload(kw_list = pesquisa, cat= 0, timeframe= '2022-5-7 2023-01-07', geo='BR',gprop='')


teste = trends.interest_over_time().drop(columns='isPartial')

fig, ax1 = plt.subplots()

color = 'blue'
ax1.set_xlabel('Data')
ax1.set_ylabel('Ações da PETR4.SA (R$)', color=color)
ax1.plot(data['Close'], color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()

color = 'tab:orange'
ax2.set_ylabel('Pesquisas "Petrobras" no Google', color=color) 
ax2.plot(teste[pesquisa], color=color)
ax2.tick_params(axis='y', labelcolor=color)
plt.title('Influência das mídias na PETR4.SA')

fig.tight_layout()

plt.show()

plt.style.use('ggplot')

data['data'] = data.index
teste['data'] = teste.index
m = pd.merge(data, teste, how='inner', on='data')
r, p = pearsonr(m['Volume'], m[pesquisa[0]] )
sns.jointplot(data=m, x="Volume", y=pesquisa[0], kind="reg")
plt.title('Correlação Volume x Busca no Google')
plt.xlabel('Volume')
plt.ylabel('Busca no Google')
plt.subplots_adjust(hspace=0.6, wspace=0.15)

plt.annotate(f'R² = {r:.3f}', xy=(0.05, 0.95), xycoords='axes fraction')
plt.show()

m.rename(columns={pesquisa[0]:"Qtd busca"}, inplace=True)
correlacao = m[['Volume' ,'Close',"Qtd busca"]].corr()

sns.heatmap(correlacao, annot = True, fmt=".2f", linewidths=.6)
plt.title('Mapa de Calor - Correlação')
plt.show()

