import pandas as pd
import os
from unidecode import unidecode

def preprocessamento(caminho_csv):
    # Leitura do CSV
    df = pd.read_csv(caminho_csv)

    # Pré-processar nomes das colunas
    novos_nomes = []
    for col in df.columns:
        nome_limpo = unidecode(col).lower().strip()
        novos_nomes.append(nome_limpo)

    df.columns = novos_nomes
    return df



