#!pip install matplotlib
#!pip install seaborn
#!pip install scikit-learn

####### TEMAS #########
# ANACONDA PROMPT
# pip install jupyterthemes
# jt -l
# jt -t <nome_tema>

####### ATUALIZAR #########
# pip install --upgrade pip

####### COMENTAR #########
#CTRL + / (barra do NUMERIC)

## Importar dados
import pandas as pd
BASE = pd.read_csv("BASE.csv")
BASE.head()

## Preparar dados para treino ##

#importa modelo de treino
from sklearn.model_selection import train_test_split

y = BASE["Vendas"] #variável alvo
x = BASE[["TV", "Radio", "Jornal"]]

# random_state = Pegar sempre a mesma porção pra treino
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.3)

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# cria as inteligencias aritificiais
modelo_regressaolinear = LinearRegression()
modelo_arvoredecisao = RandomForestRegressor()

# treina as inteligencias artificias
modelo_regressaolinear.fit(x_treino, y_treino)
modelo_arvoredecisao.fit(x_treino, y_treino)

from sklearn import metrics

# criar as previsoes
previsao_regressaolinear = modelo_regressaolinear.predict(x_teste)
previsao_arvoredecisao = modelo_arvoredecisao.predict(x_teste)

# comparar os modelos
print(metrics.r2_score(y_teste, previsao_regressaolinear))
print(metrics.r2_score(y_teste, previsao_arvoredecisao))

nova_tabela = pd.read_csv("novos.csv")
display(nova_tabela)
previsao = modelo_arvoredecisao.predict(nova_tabela)
print(previsao)

import seaborn as sns
import matplotlib.pyplot as plt

tabela_auxiliar = pd.DataFrame()
tabela_auxiliar["y_teste"] = y_teste
tabela_auxiliar["Previsoes ArvoreDecisao"] = previsao_arvoredecisao
tabela_auxiliar["Previsoes Regressao Linear"] = previsao_regressaolinear

plt.figure(figsize=(15,6))
sns.lineplot(data=tabela_auxiliar)
plt.show()

sns.heatmap(tabela_auxiliar.corr(), annot=True, cmap="Wistia")
plt.show()

sns.pairplot(tabela_auxiliar)
plt.show()