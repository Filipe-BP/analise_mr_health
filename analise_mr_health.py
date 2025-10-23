# ==========================================
# Projeto: Previsão de Demanda - MR. HEALTH
# Empresa: DataLakers
# Autor: Filipe Bernardo Pereira 
# ==========================================
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Configurações gerais
base_dir = os.path.dirname(__file__)
dados_dir = os.path.join(base_dir, "dados")
output_dir = os.path.join(base_dir, "output")

os.makedirs(output_dir, exist_ok=True)
plt.style.use('seaborn-v0_8-whitegrid')

# ---------------- 1. Leitura e integração ---------------- #
print("Lendo arquivos...")

df_pedido = pd.read_excel(os.path.join(dados_dir, 'PEDIDO-_1_.xlsx'))
df_item_pedido = pd.read_excel(os.path.join(dados_dir, 'ITEM_PEDIDO-_2_.xlsx'))
df_itens = pd.read_excel(os.path.join(dados_dir, 'ITENS-_3_.xlsx'))

df_pedido.columns = ['id', 'id_pedido', 'data', 'valor_total'][:len(df_pedido.columns)]
df_item_pedido.columns = ['id', 'id_pedido', 'id_item', 'quantidade'][:len(df_item_pedido.columns)]
df_itens.columns = ['id_item', 'valor_item'][:len(df_itens.columns)]

df_pedido = df_pedido[['id_pedido', 'data', 'valor_total']]
df_item_pedido = df_item_pedido[['id_pedido', 'id_item', 'quantidade']]
df_itens = df_itens[['id_item', 'valor_item']]

df = pd.merge(df_pedido, df_item_pedido, on='id_pedido', how='inner')
df = pd.merge(df, df_itens, on='id_item', how='inner')
df['data'] = pd.to_datetime(df['data'])

df['valor_total'] = df['quantidade'] * df['valor_item']

saida_final_path = os.path.join(output_dir, 'saida_final.xlsx')
df.to_excel(saida_final_path, index=False)
print(f"Bases integradas e salvas em '{saida_final_path}'.")

# ---------------- 2. Análise Exploratória ---------------- #
print("\n--- Análise Exploratória ---")
print(df.info())
print("\nValores nulos:\n", df.isnull().sum())
print("\nEstatísticas descritivas:\n", df.describe())

# Distribuição da quantidade vendida
plt.figure(figsize=(8, 4))
df['quantidade'].hist(bins=20)
plt.title('Distribuição da Quantidade Vendida')
plt.xlabel('Quantidade')
plt.ylabel('Frequência')
plt.show()

# Evolução temporal das vendas totais
plt.figure(figsize=(10, 4))
df.groupby('data')['quantidade'].sum().plot()
plt.title('Evolução das Vendas ao Longo do Tempo')
plt.xlabel('Data')
plt.ylabel('Quantidade Total Vendida')
plt.show()

# Top 10 produtos mais vendidos
plt.figure(figsize=(8, 4))
df.groupby('id_item')['quantidade'].sum().sort_values(ascending=False).head(10).plot(kind='bar')
plt.title('Top 10 Itens Mais Vendidos')
plt.xlabel('ID do Item')
plt.ylabel('Quantidade Total')
plt.show()

# ---------------- 3. Features ---------------- #
print("\n--- Criando features ---")
df['dias_desde_inicio'] = (df['data'] - df['data'].min()).dt.days

# Codifica o id_item
le = LabelEncoder()
df['id_item_cod'] = le.fit_transform(df['id_item'])

feature_cols = ['dias_desde_inicio', 'id_item_cod']
target_col = 'quantidade'

# Split temporal
split_idx = int(len(df) * 0.8)
train_df = df.iloc[:split_idx]
test_df = df.iloc[split_idx:]

X_train, y_train = train_df[feature_cols], train_df[target_col]
X_test, y_test = test_df[feature_cols], test_df[target_col]

# ---------------- 4. Modelo Linear ---------------- #
print("\nTreinando regressão linear simples...")
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ---------------- 5. Avaliação ---------------- #
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n=== Desempenho do Modelo Linear ===")
print(f"MSE : {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE : {mae:.4f}")
print(f"R²  : {r2:.4f}")

# ---------------- 6. Visualização do modelo ---------------- #
plt.figure(figsize=(10,5))
plt.plot(test_df['data'], y_test.values, label='Real', marker='o')
plt.plot(test_df['data'], y_pred, label='Previsto (Linear)', marker='x')
plt.title('Previsão Linear Simples de Demanda')
plt.xlabel('Data')
plt.ylabel('Quantidade')
plt.legend()
plt.show()

# ---------------- 7. Exportação ---------------- #
result = test_df.copy()
result['Prev_Linear'] = y_pred
previsoes_path = os.path.join(output_dir, 'previsoes_modelo_linear.xlsx')
result.to_excel(previsoes_path, index=False)
print(f"\nArquivo '{previsoes_path}' salvo com sucesso.")
