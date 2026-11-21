# Coloca as features na mesma escala (z-score) e depois desfaz a conta.
# Sem isso, uma feature com números grandes pode dominar a outra.

import math


def ajustar_zscore(X):
    """Normaliza cada feature para média 0 e desvio padrão 1.

    Devolve (X_normalizado, medias, desvios). Usa o desvio populacional
    (divide por n) e levanta erro se alguma feature for constante.
    """
    if not X:
        raise ValueError("não há dados para normalizar")

    dimensao = len(X[0])
    if any(len(ponto) != dimensao for ponto in X):
        raise ValueError("todos os pontos devem ter a mesma quantidade de features")

    total = len(X)
    # média e desvio de cada coluna
    medias = [sum(ponto[j] for ponto in X) / total for j in range(dimensao)]
    desvios = [
        math.sqrt(sum((ponto[j] - medias[j]) ** 2 for ponto in X) / total)
        for j in range(dimensao)
    ]

    # se a coluna é constante, não dá para dividir por zero
    if any(desvio == 0 for desvio in desvios):
        raise ValueError(
            "não é possível normalizar uma feature constante (desvio zero)"
        )

    X_normalizado = [
        [(ponto[j] - medias[j]) / desvios[j] for j in range(dimensao)] for ponto in X
    ]
    return X_normalizado, medias, desvios


def voltar_escala(w, b, medias, desvios):
    """Converte w e b do espaço normalizado de volta para a escala original."""
    if not (len(w) == len(medias) == len(desvios)):
        raise ValueError("w, médias e desvios devem ter o mesmo tamanho")

    # desfaz a divisão feita na normalização
    w_original = [wj / desvio for wj, desvio in zip(w, desvios)]
    b_original = b - sum(
        wj * media / desvio for wj, media, desvio in zip(w, medias, desvios)
    )
    return w_original, b_original
