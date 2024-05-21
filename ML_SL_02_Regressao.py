import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import VotingRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np

#0_Extraindo os Dados
df = pd.read_csv('dados/boston_house_prices.csv')

#Exibindo info dos Dados
print("\nInformações do Dados:")
print(df.info(),'\n')

print("\nInformações do Dados:")
print(df.head(),'\n')

#Removendo as colunas não úteis para a análise
df = df.drop(['AGE'], axis=1)
df = df.drop(['B'], axis=1)

#Variáveis independentes (features)
X = df.drop('MEDV', axis=1) #Incluindo Todas as colunas exceto a target

# Variável dependente (target)
y = df['MEDV'] # incluindo a taget

#2_Separando od dados em conjuntos de treinamento e teste
(X_train, X_test, y_train, y_test) = train_test_split(X,y, test_size = .20)

#Mostrando a coorelaçao entre as variaveis em um heatmap
sns.heatmap(df.corr(),annot=True, vmin=0, vmax=1)
plt.show()

#3_Dimensione os features usando StandardScaler
scaler = StandardScaler() #criando um objeto do StandardScaler()
X_train = scaler.fit_transform(X_train) #Transformar os dados e padroniza-los.
X_test = scaler.transform(X_test) #Transformar os dados

print('\nUsando os Algortimos de Machine Learning\n')

# Definindo uma lista de modelos
model_list = [
    LinearRegression(),
    KNeighborsRegressor(n_neighbors=3, p=2, metric='euclidean'),
    DecisionTreeRegressor(),
    SVR(kernel='linear'),
    RandomForestRegressor(n_estimators=10),
    GradientBoostingRegressor(random_state=1),
    VotingRegressor(estimators=[('gb', GradientBoostingRegressor(random_state=1)),
                                 ('rf', RandomForestRegressor(n_estimators=10)),
                                 ('lr', LinearRegression())])
]

# Loop sobre os modelos
for model in model_list:
    model.fit(X_train, y_train)  # Treinando o modelo
    y_pred = model.predict(X_test)  # Fazendo previsões nos dados de teste

    # Imprimindo o nome do modelo
    print(f'{type(model).__name__}')

    # Calculando métricas
    print(f'Cross Val : {cross_val_score(model, X_train, y_train, cv=5).mean():.2f}')
    print(f'R2        : {r2_score(y_test, y_pred):.2f}')
    print(f'MSE       : {mean_squared_error(y_test, y_pred):.2f}')
    print(f'RMSE      : {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}')
    print(f'MAE       : {mean_absolute_error(y_test, y_pred):.2f}\n')



