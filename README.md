# Projeto: Previsão de Demanda - MR. HEALTH

## Descrição
Este projeto tem como objetivo realizar a previsão de demanda de produtos com base em dados históricos de vendas. Ele utiliza técnicas de análise exploratória de dados (EDA) e um modelo de regressão linear simples para prever a quantidade de itens vendidos.

## Estrutura do Código
O código está dividido em várias etapas, cada uma com uma funcionalidade específica:

### 1. Leitura e Integração dos Dados
- **Arquivos de entrada:**
  - `PEDIDO-_1_.xlsx`: Contém informações sobre os pedidos, como `id_pedido`, `data` e `valor_total`.
  - `ITEM_PEDIDO-_2_.xlsx`: Contém informações sobre os itens vendidos em cada pedido, como `id_pedido`, `id_item` e `quantidade`.
  - `ITENS-_3_.xlsx`: Contém informações sobre os itens, como `id_item` e `valor_item`.
- **Processo:**
  - Os arquivos são lidos e integrados em um único DataFrame (`df`) utilizando joins (`pd.merge`).
  - A coluna `valor_total` é recalculada como o produto de `quantidade` e `valor_item`.
  - O DataFrame final é salvo como `saida_final.xlsx`.

### 2. Análise Exploratória de Dados (EDA)
- **Informações gerais:**
  - Exibe o resumo do DataFrame (`df.info()`).
  - Mostra a quantidade de valores nulos por coluna.
  - Apresenta estatísticas descritivas das colunas numéricas.
- **Visualizações:**
  - Histograma da distribuição da quantidade vendida.
  - Gráfico de linha mostrando a evolução temporal das vendas totais.
  - Gráfico de barras com os itens mais vendidos.

### 3. Criação de Features
- **Features criadas:**
  - `dias_desde_inicio`: Número de dias desde a primeira data registrada.
  - `id_item_cod`: Codificação numérica para o `id_item` usando `LabelEncoder`.
- **Divisão dos dados:**
  - Os dados são divididos em conjuntos de treinamento (80%) e teste (20%) com base no tempo.

### 4. Treinamento do Modelo
- **Modelo utilizado:**
  - Regressão Linear (`LinearRegression` do scikit-learn).
- **Variáveis:**
  - **Independentes (X):** `dias_desde_inicio` e `id_item_cod`.
  - **Dependente (y):** `quantidade`.
- **Treinamento:**
  - O modelo é treinado com os dados de treinamento.

### 5. Avaliação do Modelo
- **Métricas calculadas:**
  - **MSE (Erro Quadrático Médio):** Mede o erro médio ao quadrado entre os valores reais e previstos.
  - **RMSE (Raiz do Erro Quadrático Médio):** Raiz quadrada do MSE, interpretável na mesma escala da variável dependente.
  - **MAE (Erro Médio Absoluto):** Média dos erros absolutos.
  - **R² (Coeficiente de Determinação):** Mede o quanto o modelo explica a variação nos dados (quanto mais próximo de 1, melhor).

### 6. Visualização do Modelo
- **Gráfico gerado:**
  - Gráfico de linha comparando os valores reais e previstos no conjunto de teste ao longo do tempo.

### 7. Exportação dos Resultados
- **Arquivo gerado:**
  - `previsoes_modelo_linear.xlsx`: Contém os dados do conjunto de teste com uma nova coluna `Prev_Linear`, que armazena as previsões do modelo.

## Requisitos
- Python 3.8+
- Bibliotecas:
  - pandas
  - numpy
  - matplotlib
  - scikit-learn

## Como Executar
1. Certifique-se de que os arquivos de entrada (`PEDIDO-_1_.xlsx`, `ITEM_PEDIDO-_2_.xlsx`, `ITENS-_3_.xlsx`) estão na pasta dados.
2. Instale as dependências necessárias:
   ```bash
   pip install -r requirements.txt
   ```
3. Execute o script:
   ```bash
   python analise_mr_health.py
   ```
4. Verifique os arquivos gerados:
   - `saida_final.xlsx`: DataFrame integrado com os cálculos de `valor_total`.
   - `previsoes_modelo_linear.xlsx`: Resultados do modelo de regressão linear.

## Resultados Esperados
- **Análise Exploratória:** Gráficos e estatísticas que ajudam a entender os dados.
- **Modelo de Previsão:** Um modelo simples que prevê a quantidade vendida com base nas features criadas.
- **Arquivos de Saída:** Dados integrados e previsões salvas em arquivos Excel.

## Contato
- **Autor:** Filipe Bernardo Pereira
- **Empresa:** DataLakers