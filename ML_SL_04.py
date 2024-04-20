import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#https://www.kaggle.com/code/prabhats/linear-regression-on-house-price/input

#0_Extraindo os Dados
df = pd.read_csv('dados/housing_simple.csv')

df = df.drop(['id'], axis=1) #Removendo Colunas Desnecessaria

#convertendo a tabela date para ano
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df = df.drop(['date'], axis=1) #Removendo Date que nao e necessario

#Exibindo info dos Dados
print("\nInformações do Dados:")
print(df.info())

print("\nExibindo Parte Dados:")
print(df.head(5).to_string())

#Realizando analise Exploratoria de dados
print("\nSeparando os Dados em Numericos e Categoricos:\n")

#Encontrando e armazenando as Colunas Numericas
numerical=[]
for col in df.columns:
    if(df[col].dtypes!='object'):
        numerical.append(col)
print(f"Há um total de {len(numerical)} colunas numéricas no dataset")
print(numerical)

#Encontrando e aramazenando as Colunas Categóricas
categorical=[]
for col in df.columns:
    if(df[col].dtypes=='object'):
        categorical.append(col)
print(f"\nHá um total de {len(categorical)} colunas categóricas no dataset")
print(categorical)

#Plotando as Colunas Categóricas em histograma
df[numerical].hist(bins=15, figsize=(40, 20), layout=(5, 4))
plt.subplots_adjust(hspace=0.3, wspace=0.9)
plt.show()

# Plotando o scatterplot para ver a relacao entre variaveis
#Analisando a relaçao entre (sqft) e o preço das casas

subset_variables = ['sqft_living', 'sqft_lot',    #Selecionando um subconjunto de variáveis
                    'sqft_above', 'sqft_basement',
                    'sqft_living15', 'yr_built']

fig, axes = plt.subplots(2, 3, figsize=(20, 10)) # Configurando o layout dos subplots

# Iterando sobre cada variável em subset_variables para plotar os scatterplots
for i, variable in enumerate(subset_variables):
    row = i // 3
    col = i % 3
    sns.scatterplot(data=df, x=variable, y='price', ax=axes[row, col])
    axes[row, col].set_xlabel(variable)
    axes[row, col].set_ylabel('Price')

# Ajustando o layout
plt.tight_layout()
plt.show()

#Limpeza de Dados
print("\nRealizado a Limpeza de Dados\n")

#1)Removendo valores ausentes
print("1 - Removendo Valores Ausentes")
print('Total de Valores Ausentes: ',df.isnull().sum().sum())
print('Processando -------------------')
#Elimine as linhas com valores ausentes
df = df.dropna()
print('Total de Valores Ausentes: ',df.isnull().sum().sum())

#2)Removendo Valores Duplicado
print("\n2 - Removendo Valores Duplicados")
print('Total de Valores Duplicados: ',df.duplicated().sum())
print('Processando -----------------------')
#Removendo as duplicatas
df = df.drop_duplicates()
print('Total de Valores Duplicados: ',df.duplicated().sum())

#3)Removendo Outliers
print("\n5 - Removendo Outliers")
print('Total de Linhas: ', df.shape[0])
outlier = df.shape[0]

#Usando cálculos de mediana e IQR, os valores discrepantes são
# identificados e esses pontos de dados devem ser removidos
for coluna in df.iloc[:, :4].columns:
    Q1 = df[coluna].quantile(0.25)
    Q3 = df[coluna].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[df[coluna].between(lower_bound, upper_bound)]

print('Total Outliers: ', outlier - df.shape[0])

#1_Machine Learning - Separando os Dados e os mapeando

# Variáveis independentes (features)
X = df.drop('price', axis=1) #Incluindo Todas as colunas exceto a target

# Variável dependente (target)
y = df['price'] # incluindo a taget

#2_Separando od dados em conjuntos de treinamento e teste
(X_train, X_test, y_train, y_test) = train_test_split(X,y, test_size = .20)

print('\nInfomacoes das Dados')
print(f'Tamanho Total dos Dados : {X.shape[0]}')
print(f"Tamanho Dados de Treino : {X_train.shape[0]} Por: {X_train.shape[0]/X.shape[0]*100:.2f}%")
print(f"Tamanho Dados de Teste  : {X_test.shape[0]}  Por: {X_test.shape[0]/X.shape[0]*100:.2f}%\n")

#3_Dimensione os features usando StandardScaler
scaler = StandardScaler() #criando um objeto do StandardScaler()
X_train = scaler.fit_transform(X_train) #Transformar os dados e padroniza-los.
X_test = scaler.transform(X_test) #Transformar os dados

def metricas (y_pred,modelo,X_train, y_train):
    print(f'Cross Val : {cross_val_score(modelo, X_train, y_train, cv=5).mean():.2f}')
    print(f'R2        : {r2_score(y_test, y_pred):.2f}')
    print(f'MSE       : {mean_squared_error(y_test, y_pred):.2f}')
    print(f'RMSE      : {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}')
    print(f'MAE       : {mean_absolute_error(y_test, y_pred):.2f}\n')

#4_Usando o Algoritmo GradientBoostingRegressor
gbreg = GradientBoostingRegressor(random_state=1) # Criando uma instância do  GradientBoost
gbreg.fit(X_train, y_train) # Treinando o SVM
y_pred = gbreg.predict(X_test) # Fazendo previsões nos dados de teste
print(f'Gradient Boosting')
metricas (y_pred,gbreg,X_train, y_train)

