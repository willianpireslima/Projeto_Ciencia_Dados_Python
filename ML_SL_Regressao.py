import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn import svm
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import VotingRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import math
import numpy as np

#0_Extraindo os Dados
df = pd.read_csv('dados/boston_house_prices.csv')

# Variáveis independentes (features)
X = df.drop('MEDV', axis=1) #Incluindo Todas as colunas exceto a target

# Variável dependente (target)
y = df['MEDV'] # incluindo a taget

#Mostrando a coorelaçao entre as variaveis em um heatmap
sns.heatmap(df.corr(),annot=True, vmin=0, vmax=1)
plt.show()

#2_Separando od dados em conjuntos de treinamento e teste
(X_train, X_test, y_train, y_test) = train_test_split(X,y, test_size = .20)

print('Infomacoes das Dados')
print(f'Tamanho Total dos Dados : {X.shape[0]}')
print(f"Tamanho Dados de Treino : {X_train.shape[0]} Por: {X_train.shape[0]/X.shape[0]*100:.2f}%")
print(f"Tamanho Dados de Teste  : {X_test.shape[0]}  Por: {X_test.shape[0]/X.shape[0]*100:.2f}%\n")

#3_Dimensione os features usando StandardScaler
scaler = StandardScaler() #criando um objeto do StandardScaler()
X_train = scaler.fit_transform(X_train) #Transformar os dados e padroniza-los.
X_test = scaler.transform(X_test) #Transformar os dados

#possui y_pred é y tem quatidades diferntes de amostras é para que tenham mesmo numero limitamos 
# em y com o comando 'y[0:y_pred.shape[0]]'

def metricas (y, y_pred,modelo,X_train, y_train):
    print(f'Cross Val : {cross_val_score(modelo, X_train, y_train, cv=5).mean():.2f}')
    print(f'RSS       : {r2_score(y[0:y_pred.shape[0]], y_pred):.2f}')
    print(f'MSE       : {mean_squared_error(y[0:y_pred.shape[0]], y_pred):.2f}')
    print(f'RMSE      : {np.sqrt(mean_squared_error(y[0:y_pred.shape[0]], y_pred)):.2f}')
    print(f'MAE       : {mean_absolute_error(y[0:y_pred.shape[0]], y_pred):.2f}\n')

#4_Usando o Algoritmo LinearRegression
lnreg = linear_model.LinearRegression() # Criando uma instância do LinearRegression
lnreg.fit(X_train, y_train) # Treinando o SVM
y_pred = lnreg.predict(X_test) # Fazendo previsões nos dados de teste
print(f'Regressao linear')
metricas (y, y_pred,lnreg,X_train, y_train)

#5_Usando o Algoritmo k-nearest neighbors
knnreg = KNeighborsRegressor(n_neighbors=3,p=2,metric='euclidean')
knnreg.fit(X_train, y_train)
y_pred = knnreg.predict(X_test)
print(f"k-nearest neighbors")
metricas (y, y_pred,knnreg,X_train, y_train)

#6_Usando o Algoritmo Decision tree
dtm = DecisionTreeRegressor() # Cria o objeto classificador da Árvore de Decisão
dtm = dtm.fit(X_train,y_train) # Treinar classificador de árvore de decisão
y_pred = dtm.predict(X_test) #Prever a resposta do conjunto de dados de teste
print(f'Arvore de decisao')
metricas (y, y_pred,dtm,X_train, y_train)

#7_Usando o Algoritmo Support Vector Machine
clf = svm.SVR(kernel='linear') # Criando uma instância do SVM de regressao SVC
clf.fit(X_train, y_train) # Treinando o SVM
y_pred = clf.predict(X_test) # Fazendo previsões nos dados de teste
print(f'SVM')
metricas (y, y_pred,dtm,X_train, y_train)

#8_Usando o Algoritmo Random forests
rfreg = RandomForestRegressor(n_estimators=10) # Criando uma instância do Random forests
rfreg.fit(X_train, y_train) # Treinando o SVM
y_pred = rfreg.predict(X_test) # Fazendo previsões nos dados de teste
print(f'Random forests')
metricas (y, y_pred,rfreg,X_train, y_train)

#9_Usando o Algoritmo GradientBoostingRegressor
gbreg = GradientBoostingRegressor(random_state=1) # Criando uma instância do  GradientBoost
gbreg.fit(X_train, y_train) # Treinando o SVM
y_pred = gbreg.predict(X_test) # Fazendo previsões nos dados de teste
print(f'Gradient Boosting')
metricas (y, y_pred,gbreg,X_train, y_train)

#10_Usando o Voting
vtr = VotingRegressor(estimators=[('gb', gbreg), ('rf', rfreg), ('lr', lnreg)]) # Criando uma instância do  Voting
vtr.fit(X_train, y_train) # Treinando o SVM
y_pred = vtr.predict(X_test) # Fazendo previsões nos dados de teste
print(f'Voting')
metricas (y, y_pred,vtr,X_train, y_train)

