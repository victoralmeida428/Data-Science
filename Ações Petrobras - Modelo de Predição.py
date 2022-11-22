import pandas as pd
import pandas_datareader as web
import datetime as dt
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
start = dt.datetime(2015,1,1)
end = dt.datetime(2022,11,22)
data = web.DataReader('PETR4.SA', 'yahoo', start, end)
sc = StandardScaler()
sc.fit(data['Close'].values.reshape(-1, 1))
y = sc.transform(data['Close'].values.reshape(-1, 1))
tamanho_treino = int(data.shape[0]*0.6)
tamanho_teste = data.shape[0]-tamanho_treino
y_treino = y[:tamanho_treino]
y_teste = y[tamanho_treino:]
data['Data'] = data.index

def separa_dados(vetor, npassos):
  x_novo, y_novo = [],[]
  for i in range(npassos, vetor.shape[0]):
    x_novo.append(list(vetor.loc[i - npassos: i - 1]))
    y_novo.append(vetor.loc[i])
  x_novo, y_novo = np.array(x_novo), np.array(y_novo)
  return x_novo, y_novo
npasso = 10
vetor = pd.DataFrame(y_teste)[0]
vetor2 = pd.DataFrame(y_treino)[0]
xteste_novo, yteste_novo = separa_dados(vetor, npasso)
xtreino_novo, ytreino_novo = separa_dados(vetor2, npasso)
regressor = Sequential()
regressor.add(Dense(8, input_dim = npasso , kernel_initializer = 'random_uniform' ,
                    activation = 'linear', use_bias = True))
regressor.add(Dense(512, kernel_initializer = 'random_uniform',
                    activation = 'sigmoid', use_bias = True))
regressor.add(Dense(1, kernel_initializer = 'random_uniform',
                    activation = 'linear', use_bias = True ))
regressor.compile(loss = 'mean_squared_error', optimizer='RMSProp')
regressor.fit(xteste_novo, yteste_novo, epochs = 200)

y_predict = regressor.predict(xteste_novo)
y2_predict = regressor.predict(xtreino_novo)
treino = pd.DataFrame(y2_predict)[0]
treino = sc.inverse_transform(treino.values.reshape(1, -1))
treino = pd.DataFrame(treino).T[0]
resultado = pd.DataFrame(y_predict)[0]
resultado = sc.inverse_transform(resultado.values.reshape(1,-1))
resultado = pd.DataFrame(resultado).T[0]

sns.lineplot(x = 'Data', y = resultado.values, data=data[len(y_treino)+npasso:], label = 'simulado')
sns.lineplot(x = 'Data', y = 'Close', data=data, label = 'dados')
sns.lineplot(x = 'Data', y = treino.values, data = data[:tamanho_treino-npasso], label = 'Treino')

plt.ylabel('R$ - Close')
plt.title('sites Petrobras')
plt.xticks(rotation=70)
plt.show()
df = pd.DataFrame()
sns.regplot(x = 'Close', y = resultado.values, data = data[len(y_treino)+npasso:])
plt.xlabel("Teste")
plt.ylabel("Simulação")
plt.show()