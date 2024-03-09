import pandas as pd

#0_Carregando o arquivo CSV
df = pd.read_csv('dados/Dados_Brutos.csv')

print("Informacoes do Dados")
print(df.info())

print("\nRealizado a Limpeza de Dados\n")

print("1 - Removendo Valores Ausentes")

#1)Removendo valores ausentes
print('1 - Total de Valores Ausentes: ',df.isnull().sum().sum())
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

#3)Conversao para tipo correto de dados
print("\n3 - Conversao Para o Tipo Correto de Dados")
print('Tipos das de dados das Colunas: ')
print(df.dtypes)
print('\nProcessando ------------------\n')
# usando dicionário para converter colunas específicas
df = df.astype({'valor_1': int,'valor_2': float,'valor_3': int,'valor_4': float})
print('Tipos das de dados das Colunas')
print(df.dtypes)

#4)Conversao categorico para numerico
print("\n4 - Conversao Variaveis Categoricas Para Numericas")
print('Coluna Categorica: direcao ')
print(df['direcao'])
print('\nProcessando --------------------------------------------------\n')
#Para converter dados categóricos da coluna 'direcao' em dados numéricos
df = pd.get_dummies(df, columns=['direcao'])
print('Colunas Numericas de direcao ')
print(df[['direcao_leste','direcao_norte' , 'direcao_oeste' , 'direcao_sul']])

print("\n5 - Removendo Outliers")
print('Total de Linhas: ', df.shape[1])

#Usando cálculos de mediana e IQR, os valores discrepantes são
# identificados e esses pontos de dados devem ser removidos

for coluna in df.iloc[:, :4].columns:
    Q1 = df[coluna].quantile(0.25)
    Q3 = df[coluna].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[df[coluna].between(lower_bound, upper_bound)]

print('Total de Linhas Apos a Remocao dos Outliers: ', df.shape[0])
pd.set_option('display.max_columns', None)
print(df.corr())
# Salvar o DataFrame modificado em um novo arquivo CSV
df.to_csv('dados/Dados_Limpos.csv', index=False)