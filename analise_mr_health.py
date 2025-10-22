# ==========================================
# Projeto: Previsão de Demanda - MR. HEALTH
# Empresa: DataLakers
# Autor: Filipe Bernardo Pereira 
# ==========================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# Configurações gerais
os.chdir(os.path.dirname(__file__))
plt.style.use('seaborn-v0_8-whitegrid')

# ===================== 1. LEITURA E INTEGRAÇÃO ===================== #
print("Lendo arquivos...")

df_pedido = pd.read_excel('PEDIDO-_1_.xlsx')
df_item_pedido = pd.read_excel('ITEM_PEDIDO-_2_.xlsx')
df_itens = pd.read_excel('ITENS-_3_.xlsx')

df_pedido.columns = ['id', 'id_pedido', 'data', 'valor_total'][:len(df_pedido.columns)]
df_item_pedido.columns = ['id', 'id_pedido', 'id_item', 'quantidade'][:len(df_item_pedido.columns)]
df_itens.columns = ['id_item', 'valor_item'][:len(df_itens.columns)]

df_pedido = df_pedido[['id_pedido', 'data', 'valor_total']]
df_item_pedido = df_item_pedido[['id_pedido', 'id_item', 'quantidade']]
df_itens = df_itens[['id_item', 'valor_item']]

df = pd.merge(df_pedido, df_item_pedido, on='id_pedido', how='inner')
df = pd.merge(df, df_itens, on='id_item', how='inner')
df['data'] = pd.to_datetime(df['data'])

df.to_excel('saida_final.xlsx', index=False)
print("Bases integradas e salvas em 'saida_final.xlsx'.")

# ===================== 2. EDA (Exploração com Gráficos) ===================== #
print("\n--- EDA ---")
print(df.info())
print("\nValores nulos por coluna:\n", df.isnull().sum())
print("\nEstatísticas descritivas:\n", df.describe())

# Distribuição da quantidade vendida
plt.figure(figsize=(8, 4))
df['quantidade'].hist(bins=30)
plt.title('Distribuição da Quantidade Vendida')
plt.xlabel('Quantidade')
plt.ylabel('Frequência')
plt.show()

# Evolução temporal das vendas
plt.figure(figsize=(10, 4))
df.groupby('data')['quantidade'].sum().plot()
plt.title('Evolução das Vendas ao Longo do Tempo')
plt.xlabel('Data')
plt.ylabel('Quantidade Total Vendida')
plt.show()

# Top 10 produtos mais vendidos
plt.figure(figsize=(8, 4))
df.groupby('id_item')['quantidade'].sum().sort_values(ascending=False).head(10).plot(kind='bar')
plt.title('Top 10 Produtos Mais Vendidos')
plt.xlabel('ID do Item')
plt.ylabel('Quantidade Total')
plt.show()

# ===================== 3. ENGENHARIA DE FEATURES ===================== #
print("\n--- Engenharia de features ---")

df = df.sort_values(['id_item', 'data']).reset_index(drop=True)
df['mes'] = df['data'].dt.month
df['dia'] = df['data'].dt.day
df['dia_da_semana'] = df['data'].dt.dayofweek
df['is_weekend'] = df['dia_da_semana'].isin([5,6]).astype(int)
df['media_item'] = df.groupby('id_item')['quantidade'].transform('mean')
df['lag_1'] = df.groupby('id_item')['quantidade'].shift(1)
df['lag_7'] = df.groupby('id_item')['quantidade'].shift(7)
df['media_7d'] = df.groupby('id_item')['quantidade'].transform(lambda x: x.rolling(7, 1).mean().shift(1))

print("Features criadas: mes, dia_da_semana, is_weekend, media_item, lag_1, lag_7, media_7d")

# ===================== 4. SPLIT E MODELAGEM ===================== #
feature_cols = ['valor_item', 'mes', 'dia_da_semana', 'is_weekend', 'media_item', 'lag_1', 'lag_7', 'media_7d']
target_col = 'quantidade'

df = df.sort_values('data').reset_index(drop=True)
split_idx = int(len(df) * 0.8)
train_df = df.iloc[:split_idx].dropna(subset=feature_cols + [target_col])
test_df = df.iloc[split_idx:].dropna(subset=feature_cols + [target_col])

X_train, y_train = train_df[feature_cols], train_df[target_col]
X_test, y_test = test_df[feature_cols], test_df[target_col]

# Modelo RandomForest
rf = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# Tenta XGBoost
use_xgb = True
try:
    from xgboost import XGBRegressor
except Exception:
    use_xgb = False
    print("xgboost não disponível (pip install xgboost)")

if use_xgb:
    xgb = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=5, random_state=42, n_jobs=-1)
    xgb.fit(X_train, y_train)
    y_pred_xgb = xgb.predict(X_test)

# ===================== 5. MÉTRICAS ===================== #
def metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, rmse, mae, r2

mse, rmse, mae, r2 = metrics(y_test, y_pred_rf)
print(f"\nRandomForest - MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")

if use_xgb:
    mse, rmse, mae, r2 = metrics(y_test, y_pred_xgb)
    print(f"XGBoost     - MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")

# ===================== 6. FEATURE IMPORTANCES ===================== #
fi_rf = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=True)
plt.figure(figsize=(6, 4))
fi_rf.plot(kind='barh')
plt.title('Importância das Variáveis - RandomForest')
plt.xlabel('Importância')
plt.show()

# ===================== 7. GRÁFICOS DE PREVISÃO ===================== #
pred_df = test_df.copy()
pred_df['Prev_RF'] = y_pred_rf
if use_xgb:
    pred_df['Prev_XGB'] = y_pred_xgb

# Gráfico: previsão vs real (RandomForest)
plt.figure(figsize=(10, 5))
plt.plot(pred_df['data'], pred_df['quantidade'], label='Real', marker='o')
plt.plot(pred_df['data'], pred_df['Prev_RF'], label='Previsto (RF)', marker='x')
plt.title('Previsão de Demanda - RandomForest')
plt.xlabel('Data')
plt.ylabel('Quantidade')
plt.legend()
plt.show()

# Gráfico comparativo (XGBoost, se existir)
if use_xgb:
    plt.figure(figsize=(10, 5))
    plt.plot(pred_df['data'], pred_df['quantidade'], label='Real', marker='o')
    plt.plot(pred_df['data'], pred_df['Prev_XGB'], label='Previsto (XGBoost)', marker='x')
    plt.title('Previsão de Demanda - XGBoost')
    plt.xlabel('Data')
    plt.ylabel('Quantidade')
    plt.legend()
    plt.show()

# ===================== 8. EXPORTAR ===================== #
pred_df.to_excel('previsoes_mr_health_graficos.xlsx', index=False)
print("\nArquivo 'previsoes_mr_health_graficos.xlsx' salvo com sucesso.")
print("\n=== FIM ===")
