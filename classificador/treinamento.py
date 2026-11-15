from .modelo import classificar


def ajustar_pesos(x, y_real, w, b, taxa_aprendizado):
    """Corrige w e b na direção que aproxima o escore do rótulo correto."""
    sinal = 1 if y_real > 0 else -1
    novo_w = [wi + taxa_aprendizado * sinal * xi for wi, xi in zip(w, x)]
    novo_b = b + taxa_aprendizado * sinal
    return novo_w, novo_b


def treinar(X, y, taxa_aprendizado=0.1, epocas=100):
    """Treina um classificador linear pela regra do perceptron.

    Só ajusta quando erra: se o gabarito é +1 e o chute foi -1, soma uma
    fração do ponto aos pesos; se é -1 e o chute foi +1, subtrai.
    Devolve (w, b).
    """
    if len(X) != len(y):
        raise ValueError(f"tamanhos diferentes: {len(X)} pontos e {len(y)} rótulos")
    if not X:
        raise ValueError("não há dados para treinar")
    if taxa_aprendizado <= 0:
        raise ValueError("a taxa de aprendizado deve ser positiva")
    if epocas < 1:
        raise ValueError("o número de épocas deve ser pelo menos 1")

    w = [0.0] * len(X[0])
    b = 0.0

    for _ in range(epocas):
        erros = 0
        for x, y_real in zip(X, y):
            if classificar(x, w, b) != y_real:
                w, b = ajustar_pesos(x, y_real, w, b, taxa_aprendizado)
                erros += 1
        if erros == 0:
            break

    return w, b
