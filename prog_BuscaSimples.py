# importing pandas as pd
import pandas as pd
import matplotlib.pyplot as plt

#1_Passando o arquivo para o Dataframe
df = pd.read_csv('dados/data_pokemon.csv')

#2_Lendo arquivo
print('Tabela Completa\n',df)
print('\nTabela Parte de Cima\n',df.head())
print('\nTabela Parte de Baixo\n',df.tail())

#3_Info
print('\nInformacoes do DataFrame')
print(df.info())
print('Tamanho   : ',df.size)
print('Dimensoes : ',df.shape)

#4_Selecao e Busca

print ('\nColuna Nome do DataFrame \n',df['Nome'].head(10)) #10 Primeiros Elementos da coluna
print('\nDuas Colunas do DataFrame \n',df.loc[:, ['Nome', 'Tipo 1']]) #Selecao todas linhas : de duas colunas
print ('\nContagem da Coluna Velocidade       :',df['Velocidade'].count())
print ('Maior Elemento da Coluna Velocidade :',df['Velocidade'].max())
print ('Menor Elemento da Coluna Velocidade :',df['Velocidade'].min())
print ('Somatorio da Coluna Velocidade      :',df['Velocidade'].sum())
print ('Contagem Quantia Elementos Coluna   :\n',df['Tipo 1'].value_counts())
print('\nLinha  do DataFrame - 0\n',df.loc[0]) #Usando o loc para selecionar uma coluna no index 0,que retorna uma serie
print('\nLinha  do DataFrame - 1\n',df.loc[[1]])  # usando o [[]] para retornar um dataframe
print('\nLinhas do DataFrame - 2,3,4\n',df.loc[[2,3,4]])  # Selecao multipla
print('\nLinhas do DataFrame - 5:8\n',df.loc[5:8])  # Selecao de uma fatia
print('\nSelecao Ordencao por Geracao e Ataque: \n',df.sort_values(by=['Geracao','Ataque'],ascending=False).head(5))

df.set_index("Nome", inplace = True) #Alterando o Index para de outra coluna
print('\nLinha do DataFrame\n',df.loc[['Beedrill']])

#Selecao Especifica : Primeiro argumento são as linhas e o segundo as colunas a serem buscadas.
print('\nSelecao Item: ',df.loc['Beedrill','Tipo 1'])

#Selecao Booleana
print('\nSelecao Pokemon Tipo Grass: \n',df.loc[df["Tipo 1"] == 'Grass'].head(5))
print('\nSelecao Pokemon Ataque Maior que 300: \n',df.loc[df["Ataque"] > 60].head(5))
print('\nSelecao Pokemon Fire e Flying: \n',df.loc[(df["Tipo 1"] == 'Fire') & (df["Tipo 2"] == 'Flying')])

#Selecionado todas as linhas com condicional igual das linhas da coluna Lendario na coluna Tipo 1
print('\nSelecao Tipos do Lendario: \n',df.loc[df["Lendario"] == True,"Tipo 1"].head(5))
print('\nSelecao da Defesa Tipo Grama: \n',df.loc[df['Tipo 1']=='Grass','Defesa'].head(5))

print('\nSelecao de todos da Geracao 1,2,3: \n',df.loc[df['Geracao'].isin([1, 2,3])])

