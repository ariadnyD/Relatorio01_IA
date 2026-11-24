# Pipeline Completo de IA e Machine Learning para E-Commerce

**Aluna:** Ariadny Francisca Dantas Santos  
**Matrícula:** 20240024793  
**Disciplina:** DCT1401 - Inteligência Artificial / Tópicos Avançados de Inteligência Artificial A  
**Dataset:** [Indian E-Commerce Sales Analytics Dataset](https://www.kaggle.com/datasets/jatinkhandelwal112/indian-e-commerce-sales-analytics-dataset)

## Descrição do Projeto

Este repositório reúne a evolução completa de três etapas experimentais desenvolvidas sobre a mesma base de dados de e-commerce (`250.000` registros), abordando dois problemas supervisionados — **Regressão** (previsão do valor total da venda, `Total_Amount` / `TARGET`) e **Classificação Binária** (previsão de utilização de cupom, `Usou_Cupom` / `TARGET`, com prevalência de `~20,07%`):

1. **Etapa 1 — Análise Exploratória e Pré-processamento:** Exploração estatística, limpeza, engenharia/seleção de atributos (`Original_Price`, `Quantity`, `Pincode`) e geração dos conjuntos processados.
2. **Etapa 2 — Sistemas Baseados em Conhecimento e Modelos Clássicos:** Comparação entre Sistema Especialista (regras `SE-ENTÃO`), Árvores de Decisão e *Ensembles* (*Random Forest* e *Gradient Boosting*).
3. **Etapa 3 — Redes Neurais Profundas (MLP em PyTorch):** Implementação modular de arquiteturas *Multilayer Perceptron*, estudo sistemático de hiperparâmetros, persistência de modelos (`state_dict`), inferência limpa e análise de limiar de decisão.

## Estrutura do Repositório

* `data/`:
  * `raw/`: Dataset original bruto (não modificado).
  * `processed/`: Bases limpas e padronizadas (`dataset_regression.csv` e `dataset_classification.csv`).
* `notebooks/`:
  * `01_eda.ipynb` a `04_classification_analysis.ipynb`: Análise exploratória, pré-processamento e seleção de *features* (Etapa 1).
  * `05_expert_system.ipynb` a `07_ensemble.ipynb` (e complementares): Sistema Especialista, Árvores de Decisão, *Random Forest*, *Gradient Boosting* e análises comparativas (Etapa 2).
  * `10_mlp_regressao.ipynb` e `11_mlp_classificacao.ipynb`: Modelos *baseline* MLP em PyTorch para regressão e classificação (Etapa 3).
  * `12_hiperparametros.ipynb`: Estudo controlado de hiperparâmetros (largura, taxa de aprendizado, profundidade, ativação, otimizador, *batch size*, *dropout*, *weight decay* e interação 2D `lr × batch_size`).
  * `13_avaliacao_final.ipynb`: Treinamento final em múltiplas sementes, salvamento dos checkpoints `.pt`, avaliação no teste e análise de limiar de decisão.
* `src/`:
  * `models.py`: Definição modular da classe `MLP` configurável em PyTorch.
  * `train.py`: Divisão treino/validação/teste (`160k / 40k / 50k`), padronização, controle de sementes (`set_seed`) e treinamento com *early stopping*.
  * Demais utilitários das etapas anteriores (incluindo o motor de inferência do Sistema Especialista).
* `models/`: Checkpoints finais salvos da MLP (`mlp_final_regressao.pt` e `mlp_final_classificacao.pt`).
* `rules/`: Regras manuais do Sistema Especialista (`.txt`).
* `logs/`: Registros experimentais em CSV (incluindo `hiperparametros.csv` e `melhores_hiperparametros.csv`).
* `reports/`:
  * `figures/`: Gráficos gerados nas três etapas (exploração, árvores/ensembles, curvas de treino, hiperparâmetros, mapa de calor 2D e trade-off de limiar).
  * `tables/`: Tabelas exportadas com métricas comparativas.

## Como Executar

1. Clone este repositório em sua máquina local:
   ```bash
   git clone https://github.com/ariadnyD/nome-do-repositorio.git
   ```
2. Crie e ative o ambiente virtual e instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```
3. Execute os notebooks da pasta `notebooks/` na ordem numérica (`01` a `13`) para reproduzir o pré-processamento, os modelos clássicos/especialistas, o estudo de hiperparâmetros e a avaliação final.

## Inferência com os Modelos MLP Salvos

Os modelos neurais finais são persistidos via `state_dict` acompanhados de `model_config` e carregados em modo de avaliação (`model.eval()`):

```python
import torch
from src.models import MLP

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt = torch.load("models/mlp_final_classificacao.pt", map_location=device)

modelo = MLP(**ckpt["model_config"]).to(device)
modelo.load_state_dict(ckpt["state_dict"])
modelo.eval()

with torch.no_grad():
    probs = torch.sigmoid(modelo(X_novas_amostras.to(device)))
```

## Principais Resultados Consolidados

* **Etapa 1 (EDA e Pré-processamento):** Definição dos alvos `Total_Amount` (regressão) e `Usou_Cupom` (classificação, desbalanceada em ~80/20) e identificação de `Original_Price`, `Quantity` e `Pincode` como atributos centrais.
* **Etapa 2 (Sistema Especialista vs. Árvores/Ensembles):** Contraste entre a transparência das regras manuais `SE-ENTÃO` e o ganho preditivo obtido por Árvores de Decisão, *Random Forest* e *Gradient Boosting*.
* **Etapa 3 (Redes Neurais MLP em PyTorch):**
  * **Hiperparâmetros (Validação):** Na classificação, `dropout = 0.1` reduziu a perda média em **75,52%** frente ao *baseline* sem *dropout* (`0.0572` vs. `0.2336`), e `Tanh` superou `Sigmoid` e `ReLU`; na regressão, `lr = 5e-4` manteve alta estabilidade (`0.001180` na interação com `batch_size = 256`).
  * **Teste Final — Regressão (`mlp_final_regressao.pt`):** `RMSE = 0.0363`, `MAE = 0.0127` e $R^2 = 0.9987$.
  * **Teste Final — Classificação (`mlp_final_classificacao.pt`):** `Accuracy = 0.9854`, `Precision = 1.0000`, `Recall = 0.9274`, `F1-score = 0.9623`, `ROC-AUC = 0.9985` e `PR-AUC = 0.9952` no limiar `0.50`; ajustar o limiar para `0.25` eleva o `F1-score` para `0.9723` (`Recall = 0.9539`, `Precision = 0.9915`).
  * **Persistência:** Os checkpoints `.pt` recarregados reproduzem integralmente as predições originais.

## Contato

* **Email:** ariadny.dantas.716@ufrn.edu.br