#https://rasbt.github.io/mlxtend/user_guide/frequent_patterns/apriori/
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.frequent_patterns import fpmax

dataset = [['Leite', 'Cebola', 'Noz-moscada', 'Feijao', 'Ovos', 'Iogurte'],
            ['Aneto', 'Cebola', 'Noz-moscada', 'Feijao', 'Ovos', 'Iogurte'],
            ['Leite', 'Maça', 'Feijao', 'Ovos'],
            ['Leite', 'Vagem', 'Milho', 'Feijao', 'Iogurte'],
            ['Milho', 'Cebola', 'Cebola', 'Feijao', 'Sorvete', 'Ovos']]

# Convertendo para o formato correto
te = TransactionEncoder() #Convertendo o Dataframe em uma lista de listas, com Transactionencoder
te_ary = te.fit(dataset).transform(dataset) #Transformar os dados e padroniza-los.
df = pd.DataFrame(data=te_ary, columns=te.columns_) #passando os dados é a coluna para o Dataframe

print('Dataframe Noramalizado\n',df) #imprimindo o dataframe

#Algoritmo apriori
print('Algoritmo apriori\n',apriori(df, min_support=0.6, use_colnames=True)) #Algortimo para associar itens frequentes

#Selecionando e Filtrando Resultados
frequent_itemsets = apriori(df, min_support=0.6, use_colnames=True)
frequent_itemsets['tamanho'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))

print('\nConjunto de Itens Frequentes\n',frequent_itemsets)

#Algoritmo fpgrowth
print('\nAlgoritmo fpgrowth\n',fpgrowth(df, min_support=0.6, use_colnames=True)) #Algortimo para associar itens frequentes
frequent_itemsets = fpgrowth(df, min_support=0.6, use_colnames=True)

#Selecionando e Filtrando Resultados
frequent_itemsets['tamanho'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))

print('\nConjunto de Itens Frequentes\n',frequent_itemsets)

#Algoritmo FP-Max
print('\nAlgoritmo FP-Max\n',fpmax(df, min_support=0.6, use_colnames=True)) #Algortimo para associar itens frequentes
frequent_itemsets = fpmax(df, min_support=0.6, use_colnames=True)
