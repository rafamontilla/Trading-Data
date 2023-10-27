import pandas as pd
import quantstats as qs
import matplotlib.pyplot as plt


# Passo 2: Baixar os dados disponibilizados.
dados = pd.read_csv('dados_empresas.csv')

# Passo 3: Filtrar liquidez.
dados = dados[dados['volume_negociado'] > 1000000]

# Passo 4: Calcula os retornos mensais das empresas.
dados['retorno'] = dados.groupby('ticker')['preco_fechamento_ajustado'].pct_change()
dados['retorno'] = dados.groupby('ticker')['retorno'].shift(-1)

# Passo 5: Cria o ranking dos indicadores.
dados['ranking_ebit_ev'] = dados.groupby('data')['ebit_ev'].rank(ascending = False)
dados['ranking_roic'] = dados.groupby('data')['roic'].rank(ascending = False)

dados['ranking_final'] = dados['ranking_ebit_ev'] + dados['ranking_roic']
dados['ranking_final'] = dados.groupby('data')['ranking_final'].rank()

# Passo 6: Cria as carteiras.
dados = dados[dados['ranking_final'] <= 10]

# Passo 7: Calcula a rentabilidade por carteira.
rentabilidade_por_carteiras = dados.groupby('data')['retorno'].mean()
rentabilidade_por_carteiras = rentabilidade_por_carteiras.to_frame()

# Passo 8: Calcula a rentabilidade do modelo.
rentabilidade_por_carteiras['modelo'] = (1 + rentabilidade_por_carteiras['retorno']).cumprod() - 1

rentabilidade_por_carteiras = rentabilidade_por_carteiras.shift(1)
rentabilidade_por_carteiras = rentabilidade_por_carteiras.dropna()

# Passo 9: Calcula a rentabilidade do Ibovespa no mesmo período.
ibov = pd.read_csv('ibov.csv')
retornos_ibov = ibov['fechamento'].pct_change().dropna()
retornos_ibov_acum = (1 + retornos_ibov).cumprod() - 1
rentabilidade_por_carteiras['ibovespa'] = retornos_ibov_acum.values

rentabilidade_por_carteiras = rentabilidade_por_carteiras.drop('retorno', axis = 1)

# Passo 10: Analisar os resultados.
qs.extend_pandas()
rentabilidade_por_carteiras.index = pd.to_datetime(rentabilidade_por_carteiras.index)

rentabilidade_por_carteiras['modelo'].plot_monthly_heatmap()
rentabilidade_por_carteiras['ibovespa'].plot_monthly_heatmap()

rentabilidade_por_carteiras.plot()

rentabilidade_ao_ano = (1 + rentabilidade_por_carteiras.loc['2023-06-30', 'modelo']) ** (1/10.66) - 1

print(rentabilidade_ao_ano)

