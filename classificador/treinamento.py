import random
from dataclasses import dataclass

from .modelo import classificar


@dataclass
class ResultadoPerceptron:
    """Resultado do treinamento pela regra do perceptron."""

    w: list
    b: float
    convergiu: bool
    epocas: int
    erros_por_epoca: list


def ajustar_pesos(x, y_real, w, b, taxa_aprendizado):
    """Corrige w e b na direção que aproxima o escore do rótulo correto."""
    sinal = 1 if y_real > 0 else -1
    novo_w = [wi + taxa_aprendizado * sinal * xi for wi, xi in zip(w, x)]
    novo_b = b + taxa_aprendizado * sinal
    return novo_w, novo_b


def treinar(
    X,
    y,
    taxa_aprendizado=0.01,
    epocas=100,
    embaralhar=True,
    semente=42,
):
    """Treina um classificador linear pela regra do perceptron.

    Só ajusta quando erra. Se ``embaralhar`` for verdadeiro, a ordem dos
    pontos muda a cada época usando ``semente`` para reprodutibilidade.
    Devolve um ResultadoPerceptron com a flag ``convergiu``.
    """
    if len(X) != len(y):
        raise ValueError(f"tamanhos diferentes: {len(X)} pontos e {len(y)} rótulos")
    if not X:
        raise ValueError("não há dados para treinar")
    if taxa_aprendizado <= 0:
        raise ValueError("a taxa de aprendizado deve ser positiva")
    if epocas < 1:
        raise ValueError("o número de épocas deve ser pelo menos 1")

    rng = random.Random(semente)
    w = [0.0] * len(X[0])
    b = 0.0
    erros_por_epoca = []
    convergiu = False
    epocas_executadas = 0

    for _ in range(epocas):
        epocas_executadas += 1
        indices = list(range(len(X)))
        if embaralhar:
            rng.shuffle(indices)

        erros = 0
        for indice in indices:
            if classificar(X[indice], w, b) != y[indice]:
                w, b = ajustar_pesos(X[indice], y[indice], w, b, taxa_aprendizado)
                erros += 1

        erros_por_epoca.append(erros)
        if erros == 0:
            convergiu = True
            break

    return ResultadoPerceptron(
        w=w,
        b=b,
        convergiu=convergiu,
        epocas=epocas_executadas,
        erros_por_epoca=erros_por_epoca,
    )
