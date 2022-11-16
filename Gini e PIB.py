import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
def modelagem(arquivo):
    arquivo = arquivo.T
    arquivo.ffill(inplace=True)
    arquivo = arquivo.T
    ano = []
    for i in range(1960, 2021, 1):
        ano.append(str(i))
    arquivo.drop(columns=ano, inplace=True)
    arquivo.drop(columns=['Indicator Name', 'Indicator Code'], inplace=True)
    arquivo.set_index('Country Code', inplace=True)
    return arquivo
#TRATAMENTO DE DADOS ÍNDICE GINI
gini = pd.read_csv('API_SI.POV.GINI_DS2_en_csv_v2_4701295.csv', sep=',')
gini = modelagem(gini)
gini['2021'] = gini['2021'].replace('SI.POV.GINI',0)
gini.rename(columns={'2021':'Gini'}, inplace=True)

#TRATAMENTO DE DADOS ÍNDICE PIB PER CAPITA
pibpc = pd.read_csv('API_NY.GDP.PCAP.CD_DS2_en_csv_v2_4701206.csv',sep=',')
pibpc=modelagem(pibpc)
pibpc.rename(columns={'2021':'PIB per Capita'}, inplace=True)
pibpc['PIB per Capita'] = pibpc['PIB per Capita'].replace('NY.GDP.PCAP.CD',0)

#MESCLANDO OS DATAFRAMES
dados = pd.merge(gini, pibpc, how='inner')
dados.set_index(dados['Country Name'],inplace=True)
dados.drop(columns=['Country Name'], inplace=True)
filtro = (dados['PIB per Capita'] > 0) & (dados['Gini'] > 0)
ajuste = dados.loc[~filtro]
dados.drop(ajuste.index, inplace=True)
r2 = dados['PIB per Capita'].corr(dados['Gini'])
texto = 'R²: ' + str(r2.round(3))

#Construção do Gráfico
sns.regplot(dados, x=dados['Gini'],y=dados['PIB per Capita'])
plt.title("Relação PIB per Capita x Índice Gini")
plt.xlabel('Gini')
plt.ylabel('PIB per Capita (US$)')
plt.text(x=55, y=120000, s=texto)
plt.show()
