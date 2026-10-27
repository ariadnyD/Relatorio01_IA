import pandas as pd
import numpy as np

class BaseDeConhecimento:
    """Armazena as regras do Sistema Especialista (extraídas dos .txt)"""
    
    @staticmethod
    def regra_classificacao(linha):
        # Regra 1 (Heurística)
        if linha['Total_Amount'] > 200 and linha['Total_Spent'] > 1000:
            return 1
        # Regra 2 (Relação)
        elif linha['Customer_Age'] < 30 and linha['Order_Value'] > 150:
            return 1
        # Regra 3 (Estratégia)
        elif linha['Total_Amount'] <= 50 and linha['Customer_Age'] >= 50:
            return 0
        return 0 # Default

    @staticmethod
    def regra_regressao(linha):
        # Regra 1 (Relação)
        if linha['Original_Price'] > 500 and linha['Quantity'] >= 3:
            return 1500.0
        # Regra 2 (Heurística)
        elif linha['Discount_Percent'] > 20 and linha['Quantity'] == 1:
            return 100.0
        # Regra 3 (Diretriz)
        elif linha['Original_Price'] <= 100 and linha['Quantity'] == 2:
            return 180.0
        # Default: heurística simples (preço * quantidade)
        return linha['Original_Price'] * linha['Quantity']

class MotorInferencia:
    """Aplica a base de conhecimento aos dados usando Encadeamento para Frente"""
    def __init__(self, tipo='classificacao'):
        self.tipo = tipo
        self.base = BaseDeConhecimento()

    def inferir_amostra(self, linha):
        if self.tipo == 'classificacao':
            return self.base.regra_classificacao(linha)
        else:
            return self.base.regra_regressao(linha)

    def prever(self, X):
        return X.apply(self.inferir_amostra, axis=1)