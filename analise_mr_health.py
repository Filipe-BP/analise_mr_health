import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df_pedido= pd.read_excel('PEDIDO-_1_.xlsx')
df_item_pedido= pd.read_excel('ITEM_PEDIDO-_2_.xlsx')
df_itens= pd.read_excel('ITENS-_3_.xlsx')


novos_nomes_df_pedido = ['id', 'id_pedido', 'data', 'valor_total']
k = min(len(novos_nomes_df_pedido), len(df_pedido.columns))
cols = df_pedido.columns.tolist()
cols[:k] = novos_nomes_df_pedido[:k]
df_pedido.columns = cols

novos_nomes_df_item_pedido = ['id', 'id_pedido', 'id_item', 'quantidade']
k = min(len(novos_nomes_df_item_pedido), len(df_item_pedido.columns))
cols = df_item_pedido.columns.tolist()
cols[:k] = novos_nomes_df_item_pedido[:k]
df_item_pedido.columns = cols

novos_nomes_df_itens = ['id_item', 'valor_item']
k = min(len(novos_nomes_df_itens), len(df_itens.columns))
cols = df_itens.columns.tolist()
cols[:k] = novos_nomes_df_itens[:k]
df_itens.columns = cols


colunas_pedido = ['id_pedido', 'data', 'valor_total']
colunas_item_pedido = ['id_pedido', 'id_item', 'quantidade']
colunas_itens = ['id_item', 'valor_item']

df_pedido = df_pedido[colunas_pedido]
df_item_pedido = df_item_pedido[colunas_item_pedido]
df_itens = df_itens[colunas_itens]


pedido_item = pd.merge(df_pedido, df_item_pedido, on='id_pedido', how='inner')


df_final = pd.merge(pedido_item, df_itens, on='id_item', how='inner')


print(df_final.head())

df_final.to_excel('saida_final.xlsx', index=False)


###########################################################################################################
# Preparação dos dados para previsão de demanda
# Selecionar as colunas relevantes
X = df_final[['id_pedido', 'id_item', 'valor_item']]  # Variáveis independentes
X = pd.get_dummies(X, columns=['id_pedido', 'id_item'], drop_first=True)  # Codificação one-hot

y = df_final['quantidade']  # Variável dependente

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinar o modelo de regressão linear
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# Fazer previsões no conjunto de teste
y_pred = modelo.predict(X_test)

# Avaliar o desempenho do modelo
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Erro quadrático médio (MSE): {mse}")
print(f"Coeficiente de determinação (R²): {r2}")

# Exibir os coeficientes e o intercepto do modelo
print("Coeficientes do modelo:")
for feature, coef in zip(X.columns, modelo.coef_):
    print(f"{feature}: {coef}")

print(f"Intercepto: {modelo.intercept_}")

# Exibir as previsões individuais para o conjunto de teste
print("\nPrevisões para o conjunto de teste:")
previsoes = pd.DataFrame({
    'Real': y_test,
    'Previsto': y_pred
})
print(previsoes.head())
