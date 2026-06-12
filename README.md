# Previsão da Inflação em Portugal — Modelos de Machine Learning

Projeto Final Aplicado em Ciência de Dados · ISCTE 2025/2026 · **Grupo 11**
Cliente: **Banco de Portugal** · Orientadora: Prof.ª Diana Mendes

Previsão da taxa de inflação homóloga portuguesa a partir de variáveis macroeconómicas nacionais, europeias e norte-americanas, comparando modelos econométricos clássicos (ARIMA, VAR, CC-VAR) com modelos de Machine Learning (Ridge, Random Forest, LightGBM).

**Dashboard:** https://inflationforecast.netlify.app/dashboard

## Instalação

Criar e ativar um ambiente virtual e instalar as dependências a partir da raiz do projeto:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Se usar VS Code ou Jupyter, selecionar o interpretador Python do ambiente `.venv`. Correr os notebooks a partir da raiz do projeto, para que os caminhos relativos para dados, modelos e scripts auxiliares resolvam corretamente.

## Estrutura do projeto

| Pasta/Ficheiro | Conteúdo |
|---|---|
| `dados/` | Fontes originais e datasets processados (treino, teste, dataset consolidado) |
| `models/` | Modelos finais treinados (`.pkl`) e transformadores de pré-processamento |
| `Imagens/` | ALgumas figuras geradas pelos notebooks |
| `pipeline_datapreparation.py` | Pipeline de preparação (interpolação, ragged edges, estacionarização, outliers, lags, scaling) |
| `FinalPipeline.py` | Classe `InflationModel` para produção (fit/predict/forecast) |
| `routine_data.py` | Utilitários de leitura de dados |
| `find_best_seed_CC_VAR.py` | Procura de seed para o CC-VAR |
| `requirements.txt` | Dependências Python |

## Sequência de notebooks

Os notebooks estão numerados pela ordem do fluxo de trabalho, da preparação dos dados à avaliação final:

| Notebook | Papel |
|---|---|
| `01_DataSourcesPreparation` | Leitura e validação das fontes de dados |
| `02_DataIntegration` | Integração das fontes num dataset unificado |
| `03_BivariateAnalysis` | Análise exploratória bivariada (correlações, lags, autocorrelação) |
| `04_TimeSeriesAnalysis` | Testes de estacionaridade (ADF, KPSS) e transformações |
| `05_DataProcessing` | Interpolação, ragged edges (Auto-ARIMA), outliers |
| `06_DataPreparation` | Seleção de variáveis (Causalidade de Granger) e de lags (Lasso) |
| `07_ARIMA_Model` | Modelo ARIMA univariado (benchmark) |
| `08_VAR_Model` | Modelo VAR multivariado |
| `09_CC_VAR_Model` | Modelo CC-VAR (PCA + VAR) |
| `10_Modeling_ML_Computacional` | Modelos ML com variáveis de seleção computacional |
| `11_Modeling_ML_Teoricas` | Modelos ML com melhores features; explicabilidade (SHAP) |
| `12_FinalModels` | Pipeline de produção, modelos finais e previsões futuras |

## Fontes de dados

Banco de Portugal/BPstat (variável alvo e taxa de câmbio EUR/USD), Euro Area EA-MD-QD (indicadores macro portugueses, mensais e trimestrais), FRED-MD (indicadores dos EUA), Índice de Incerteza de Política Económica para Portugal (Morão, 2024), OCDE (taxa de desemprego) e Índice de Preços no Produtor para Portugal. As referências completas constam do relatório final.
