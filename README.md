# Sistema Preditivo e Especialista para E-Commerce

**Aluno:** Ariadny Francisca Dantas Santos  
**Matrícula:** 20240024793  
**Disciplina:** DCT1401 - Inteligência Artificial  
**Dataset:** [Indian E-Commerce Sales Analytics Dataset](https://www.kaggle.com/datasets/jatinkhandelwal112/indian-e-commerce-sales-analytics-dataset)

## Descrição do Projeto
Este repositório contém o desenvolvimento completo de um pipeline de Inteligência Artificial e Machine Learning. O projeto compara abordagens baseadas em conhecimento (Sistemas Especialistas com regras SE-ENTÃO) com modelos baseados em dados (Árvores de Decisão e Ensembles: Random Forest e Gradient Boosting) aplicados a dois problemas distintos: **Regressão** e **Classificação**.

## Estrutura do Repositório
- `data/raw/`: Dataset original bruto
- `data/processed/`: Datasets limpos e processados (`dataset_classification.csv` e `dataset_regression.csv`)
- `notebooks/`: 
  - `01_eda.ipynb` a `04_classification_analysis.ipynb`: Análise exploratória, limpeza e seleção de *features*.
  - `05_expert_system.ipynb`: Avaliação do Sistema Especialista (Modelo 1).
  - `06_decision_tree.ipynb`: Treinamento e ajuste de Árvores de Decisão (Modelo 2).
  - `07_ensemble.ipynb`: Modelos de Ensemble (Random Forest e Gradient Boosting - Modelo 3).
  - Demais notebooks: Avaliação de métricas, generalização e eficiência computacional.
- `rules/`: Regras manuais do Sistema Especialista em formato textual (`.txt`).
- `logs/`: Registros de cronometragem do esforço de engenharia de conhecimento (`.csv`).
- `reports/figures/`: Gráficos gerados (Curvas ROC/PR, importância de *features*, resíduos e árvores).
- `reports/tables/`: Tabelas exportadas com as métricas comparativas de desempenho.
- `src/`: Módulos utilitários (incluindo o motor de inferência do Sistema Especialista).

## Como Executar
1. Clone este repositório em sua máquina local.
2. Instale as dependências executando no terminal: `pip install -r requirements.txt`
3. Certifique-se de executar os notebooks sequencialmente na pasta `notebooks/` para garantir a correta geração dos dados e modelos salvos.

## Principais Resultados
- **Classificação:** Previsão de utilização de cupons (`Usou_Cupom`), avaliada por Acurácia, F1-Score, Curvas ROC e Precision-Recall.
- **Regressão:** Previsão do valor total da venda (`Total_Amount`), avaliada por RMSE, MAE e $R^2$.
- **Análise Comparativa:** Contraste direto entre a interpretabilidade das regras manuais do Sistema Especialista e a capacidade preditiva robusta dos algoritmos de Aprendizado de Máquina.

## Contato
- Email: ariadny.dantas.716@ufrn.edu.br