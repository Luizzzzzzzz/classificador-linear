# Aqui está a conta do mínimos quadrados.
# A gente monta a matriz A, resolve o sistema normal AᵀA z = Aᵀb na mão
# (Gauss com pivoteamento) e devolve a reta, o determinante e o resíduo.

from dataclasses import dataclass


@dataclass
class ResultadoMQ:
    """Resultado do treinamento por mínimos quadrados."""

    w: list
    b: float
    determinante: float
    residuo: float


def montar_sistema(X, y):
    """Monta A (coluna de 1s + features) e b (rótulos) para A z ≈ b."""
    if len(X) != len(y):
        raise ValueError(f"tamanhos diferentes: {len(X)} pontos e {len(y)} rótulos")
    if not X:
        raise ValueError("não há dados para montar o sistema")

    dimensao = len(X[0])
    if any(len(x) != dimensao for x in X):
        raise ValueError("todos os pontos devem ter a mesma quantidade de features")

    # a primeira coluna é toda de 1s: é ela que dá o termo independente b
    A = [[1.0, *[float(valor) for valor in x]] for x in X]
    # b é o vetor com os rótulos (+1/-1) do gabarito
    b = [float(rotulo) for rotulo in y]
    return A, b


def transposta(A):
    """Devolve a transposta de uma matriz representada por listas de linhas."""
    return [list(coluna) for coluna in zip(*A)]


def multiplicar(A, B):
    """Multiplica duas matrizes representadas por listas de linhas."""
    Bt = transposta(B)
    return [[sum(a * b for a, b in zip(linha, coluna)) for coluna in Bt] for linha in A]


def multiplicar_matriz_vetor(A, v):
    """Multiplica uma matriz por um vetor coluna."""
    return [sum(a * b for a, b in zip(linha, v)) for linha in A]


def determinante(M):
    """Calcula o determinante de uma matriz quadrada por expansão em cofatores."""
    n = len(M)
    if n == 0 or any(len(linha) != n for linha in M):
        raise ValueError("o determinante exige uma matriz quadrada não vazia")

    if n == 1:
        return M[0][0]
    if n == 2:
        return M[0][0] * M[1][1] - M[0][1] * M[1][0]

    total = 0.0
    for coluna in range(n):
        menor = [linha[:coluna] + linha[coluna + 1 :] for linha in M[1:]]
        sinal = -1 if coluna % 2 else 1
        total += sinal * M[0][coluna] * determinante(menor)
    return total


def resolver_gauss(M, v, tolerancia=1e-12):
    """Resolve M z = v por eliminação de Gauss com pivoteamento parcial."""
    n = len(M)
    if n == 0 or len(v) != n or any(len(linha) != n for linha in M):
        raise ValueError("sistema inválido: M deve ser quadrada e compatível com v")

    # cópias para não mexer na matriz original
    M = [list(linha) for linha in M]
    v = list(v)

    # tolerância relativa: se a matriz tem números grandes, o limite cresce junto
    escala = max(abs(valor) for linha in M for valor in linha)
    limite = tolerancia * max(1.0, escala)

    for coluna in range(n):
        # pega o maior pivô possível para dividir por um número confortável
        pivo = max(range(coluna, n), key=lambda linha: abs(M[linha][coluna]))
        if abs(M[pivo][coluna]) < limite:
            raise ValueError("sistema singular ou mal condicionado")
        # troca a linha do pivô para a posição atual
        if pivo != coluna:
            M[coluna], M[pivo] = M[pivo], M[coluna]
            v[coluna], v[pivo] = v[pivo], v[coluna]

        # zera os valores abaixo do pivô
        for linha in range(coluna + 1, n):
            fator = M[linha][coluna] / M[coluna][coluna]
            for c in range(coluna, n):
                M[linha][c] -= fator * M[coluna][c]
            v[linha] -= fator * v[coluna]

    # retrosubstituição: de baixo para cima, acha cada valor de z
    z = [0.0] * n
    for linha in reversed(range(n)):
        soma = sum(M[linha][c] * z[c] for c in range(linha + 1, n))
        z[linha] = (v[linha] - soma) / M[linha][linha]
    return z


def treinar_minimos_quadrados(X, y):
    """Resolve AᵀA z = Aᵀb por Gauss e devolve um ResultadoMQ.

    A solução z = [b, w1, w2, ...] minimiza o erro quadrático entre A z e y.
    O resultado inclui det(AᵀA) e o resíduo ||Az − b||².
    """
    A, b = montar_sistema(X, y)
    # o sistema normal: AᵀA z = Aᵀb
    At = transposta(A)
    AtA = multiplicar(At, A)
    Atb = multiplicar_matriz_vetor(At, b)
    z = resolver_gauss(AtA, Atb)

    # resíduo: soma dos erros quadráticos que sobraram
    Az = multiplicar_matriz_vetor(A, z)
    residuo = sum((ajuste - alvo) ** 2 for ajuste, alvo in zip(Az, b))
    return ResultadoMQ(
        w=z[1:],
        b=z[0],
        determinante=determinante(AtA),
        residuo=residuo,
    )
