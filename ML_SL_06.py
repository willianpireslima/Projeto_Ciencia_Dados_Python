import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np
from sklearn import preprocessing

#https://www.kaggle.com/code/arvindkhoda/obesity-levels-analysis/input

# Lendo o arquivo CSV em um DataFrame
df = pd.read_csv('dados/obesity_levels.csv')

#Exibindo info dos Dados
print("\nInformações do Dados:")
print(df.info())

print("\nExibindo Parte Dados:")
print(df.head(5).to_string())

#Realizando Analise Exploratoria dos Dados
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

#Plotando as categoarias numericas em historiograma
df[numerical].hist(bins=15, figsize=(15, 6), layout=(2, 4))
plt.show()

#Plotando as categoricas em pizza
fig, ax = plt.subplots(3, 3, figsize=(20, 10))
# Iterando sobre as colunas categóricas e plotando um gráfico de pizza para cada uma
for variable, subplot in zip(categorical, ax.flatten()):
    pie = df[variable].value_counts().plot(kind='pie', ax=subplot, autopct='%1.1f%%', labeldistance=1.2)
    subplot.set_title(variable)
    subplot.set_ylabel('')  # Remove o rótulo do eixo y para os gráficos de pizza
    # Ajustando a distância das legendas de porcentagem dentro do gráfico de pizza
# Ajustando o layout
plt.tight_layout()
plt.show()

#Plotando a relacao entre duas variaveis
#Configurando o layout dos subplots
fig, ax = plt.subplots(3, 3, figsize=(15, 10))
for variable, subplot in zip(categorical, ax.flatten()):
    sns.boxplot(x=variable, y='NObeyesdad', data=df, ax=subplot)
    # Rotacionando as legendas do eixo x graus
    for tick in subplot.get_xticklabels():
        tick.set_rotation(25)

# Ajustando o espaçamento entre os subplots
plt.subplots_adjust(wspace=0.8, hspace=0.7)
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

#4)Conversao categorico para numerico
print("\n4 - Conversao Variaveis Categoricas Para Numericas")
print('\nProcessando --------------------------------------------------\n')

# label_encoder object knows
label_encoder = preprocessing.LabelEncoder()

#Percorrendo Todas as colunas da lista categorical gerada anteriormente
# O label_encoder esta transformando categorico para numerico
for var in categorical:
    df[var] = label_encoder.fit_transform(df[var])

print('Colunas Numericas')
print(df[categorical].head(5).to_string())

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
X = df.drop('NObeyesdad', axis=1) #Incluindo Todas as colunas exceto a target

# Variável dependente (target)
y = df['NObeyesdad'] # incluindo a taget

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

def metricas (y, y_pred,modelo,X_train, y_train):
    print(f'Cross Val : {cross_val_score(modelo, X_train, y_train, cv=5).mean():.2f}')
    print(f'R2        : {r2_score(y_test, y_pred):.2f}')
    print(f'MSE       : {mean_squared_error(y_test, y_pred):.2f}')
    print(f'RMSE      : {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}')
    print(f'MAE       : {mean_absolute_error(y_test, y_pred):.2f}\n')

#4_Usando o Algoritmo Random forests
rfreg = RandomForestRegressor(n_estimators=10) # Criando uma instância do Random forests
rfreg.fit(X_train, y_train) # Treinando o SVM
y_pred = rfreg.predict(X_test) # Fazendo previsões nos dados de teste
print(f'Random forests')
metricas (y, y_pred,rfreg,X_train, y_train)

