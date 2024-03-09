import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(8,5))

#Grafico em linha
lin = pd.read_csv('dados/data_gas_preco.csv')

plt.title('Preco do Gas ao longo do tempo na America do Norte', fontdict={'fontname': 'serif', 'fontsize': 10})
plt.xlabel('Ano')
plt.ylabel('Valor')
plt.plot(lin["Ano"],lin["EUA"],'r.--',label='EUA')
plt.plot(lin["Ano"],lin["Mexico"],'bo-.',label='Mexico')
plt.plot(lin["Ano"],lin["Canada"],'g<:',label='Canada')
plt.legend()
plt.show()

plt.title('Preco do Gas ao longo do tempo na Mundo', fontdict={'fontname': 'serif', 'fontsize': 16})
plt.xlabel('Ano')
plt.ylabel('Valor')
for nacao in lin:
    if nacao != 'Ano':
        plt.plot(lin["Ano"], lin[nacao], marker='.',label=nacao)
plt.legend()
plt.show()

#Grafico em Pizza
pizza = pd.read_csv('dados/data_populacaoue.csv') # Selecionar os valores da coluna 'População'

valores = pizza['Populacao']# Selecionar os valores da coluna 'População'
rotulos = pizza['Pais']     # Selecionar os rótulos da coluna 'País'

plt.title('Populacao Uniao Europeia em 2022', fontdict={'fontname': 'serif', 'fontsize': 16})
plt.pie(valores, labels=rotulos)
plt.show()

#Grafico em Barra
bar = pd.read_csv('dados/data_populacaoue.csv') # Selecionar os valores da coluna 'População'

valores = bar['Populacao']# Selecionar os valores da coluna 'População'
rotulos = bar['Pais']     # Selecionar os rótulos da coluna 'País'

plt.title('Populacao Uniao Europeia em 2022', fontdict={'fontname': 'serif', 'fontsize': 16})
plt.bar(rotulos,valores)
plt.show()

#Grafico em Histograma
his = pd.read_csv('dados/data_iris.csv')
plt.title('Distribuição do Comprimento da Sépala para Setosa ', fontdict={'fontname': 'serif', 'fontsize': 16})
plt.hist(his.loc[his['variedade'] == "Setosa", 'sepala.comprimento'],bins=10,alpha=0.5)
plt.show()

#Grafico heatmap no Seaborn
heat = pd.read_csv('dados/data_gorjetas.csv')
#convertendo as colunas categoricas para numericas
heat = pd.get_dummies(heat, columns=['sexo'])
heat = pd.get_dummies(heat, columns=['fumante'])
heat = pd.get_dummies(heat, columns=['dia'])
heat = pd.get_dummies(heat, columns=['hora'])
#armazenando a correlacao de dados
correlacao = heat.corr()
sns.heatmap(correlacao)
#rotacionado a legenda x para ficar mais legivel
plt.xticks(rotation=15)
plt.show()