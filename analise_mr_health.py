import pandas as pd

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
