from .modelo import classificar


def avaliar(y_real, y_previsto):
    """Compara gabarito e chutes, devolvendo total, acertos, erros e acurácia (0–1)."""
    if len(y_real) != len(y_previsto):
        raise ValueError(
            f"tamanhos diferentes: {len(y_real)} rótulos reais e {len(y_previsto)} chutes"
        )
    if not y_real:
        raise ValueError("não há dados para avaliar")

    total = len(y_real)
    acertos = sum(1 for real, chute in zip(y_real, y_previsto) if real == chute)
    return {
        "total": total,
        "acertos": acertos,
        "erros": total - acertos,
        "acuracia": acertos / total,
    }


def avaliar_modelo(X, y, w, b):
    """Classifica todos os pontos de X e compara com o gabarito y."""
    chutes = [classificar(x, w, b) for x in X]
    return avaliar(y, chutes)


def matriz_confusao(y_real, y_previsto):
    """Conta VP, VN, FP e FN considerando +1 como classe positiva."""
    if len(y_real) != len(y_previsto):
        raise ValueError(
            f"tamanhos diferentes: {len(y_real)} rótulos reais e {len(y_previsto)} chutes"
        )

    resultado = {"VP": 0, "VN": 0, "FP": 0, "FN": 0}
    for real, chute in zip(y_real, y_previsto):
        if real == 1 and chute == 1:
            resultado["VP"] += 1
        elif real == -1 and chute == -1:
            resultado["VN"] += 1
        elif real == -1 and chute == 1:
            resultado["FP"] += 1
        elif real == 1 and chute == -1:
            resultado["FN"] += 1
    return resultado


def metricas_por_limiar(y_real, escores, limiar=0.0):
    """Classifica por escore >= limiar e devolve acurácia, sensibilidade e especificidade."""
    if len(y_real) != len(escores):
        raise ValueError(
            f"tamanhos diferentes: {len(y_real)} rótulos reais e {len(escores)} escores"
        )
    if not y_real:
        raise ValueError("não há dados para avaliar")

    chutes = [1 if escore >= limiar else -1 for escore in escores]
    contagens = matriz_confusao(y_real, chutes)
    total = len(y_real)
    positivos = contagens["VP"] + contagens["FN"]
    negativos = contagens["VN"] + contagens["FP"]

    return {
        **contagens,
        "limiar": limiar,
        "acuracia": (contagens["VP"] + contagens["VN"]) / total,
        "sensibilidade": contagens["VP"] / positivos if positivos else 0.0,
        "especificidade": contagens["VN"] / negativos if negativos else 0.0,
    }


