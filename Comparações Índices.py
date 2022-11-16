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
#TRATAMENTO DE DADOS ÍNDICE GINI - WORLDBANK DATA
gini = pd.read_csv('API_SI.POV.GINI_DS2_en_csv_v2_4701295.csv', sep=',')
gini = modelagem(gini)
gini['2021'] = gini['2021'].replace('SI.POV.GINI',0)
gini.rename(columns={'2021':'Gini'}, inplace=True)

#TRATAMENTO DE DADOS ÍNDICE PIB PER CAPITA - WORLDBANK DATA
pibpc = pd.read_csv('API_NY.GDP.PCAP.CD_DS2_en_csv_v2_4701206.csv',sep=',')
pibpc=modelagem(pibpc)
pibpc.rename(columns={'2021':'PIB per Capita'}, inplace=True)
pibpc['PIB per Capita'] = pibpc['PIB per Capita'].replace('NY.GDP.PCAP.CD',0)

#TRATAMENTO ÍNDICE BETTER LIFE - OECD
blindex = pd.read_csv('BLI_16112022165706046.csv', sep=',')
blindex.rename(columns={'LOCATION':'Country Code', 'Country': 'Country Name'}, inplace=True)
blindex.set_index('Country Code', inplace=True)

#MESCLANDO OS DATAFRAMES GINI E PIB
dados = pd.merge(gini, pibpc, how='inner')
dados.set_index(dados['Country Name'],inplace=True)
dados.drop(columns=['Country Name'], inplace=True)
filtro = (dados['PIB per Capita'] > 0) & (dados['Gini'] > 0)
ajuste = dados.loc[~filtro]
dados.drop(ajuste.index, inplace=True)
r2 = dados['PIB per Capita'].corr(dados['Gini'])
texto = 'R²: ' + str(r2.round(3))

#MESCLANDO OS DATAFRAMES GINI E PIB
ginihappy = pd.merge(blindex,gini)
r2 = ginihappy['Value'].corr(ginihappy['Gini'])
texto2 = 'R²: ' + str(r2.round(3))

#Construção do Gráfico
plt.subplot(211)
sns.regplot(dados, x=dados['Gini'],y=dados['PIB per Capita'])
plt.title("Relação PIB per Capita x Índice Gini")
plt.xlabel('Gini')
plt.ylabel('PIB per Capita (US$)')
plt.xlim(dados['Gini'].min()-1,dados['Gini'].max()+1)
plt.ylim(dados['PIB per Capita'].min()-10000,dados['PIB per Capita'].max()+10000)
plt.text(x=55, y=120000, s=texto)

plt.subplot(212)
sns.regplot(ginihappy, x=ginihappy['Gini'],y=ginihappy['Value'])
plt.title("Relação Life Satisfaction x Índice Gini")
plt.xlabel('Gini')
plt.ylabel('Index Life Satisfaction')
plt.xlim(ginihappy['Gini'].min()-1,ginihappy['Gini'].max()+1)
plt.text(x=55, y=93, s=texto2)
plt.show()

