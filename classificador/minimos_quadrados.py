def montar_sistema(X, y):
    """Monta A (coluna de 1s + features) e b (rótulos) para A z ≈ b."""
    if len(X) != len(y):
        raise ValueError(f"tamanhos diferentes: {len(X)} pontos e {len(y)} rótulos")
    if not X:
        raise ValueError("não há dados para montar o sistema")

    dimensao = len(X[0])
    if any(len(x) != dimensao for x in X):
        raise ValueError("todos os pontos devem ter a mesma quantidade de features")

    A = [[1.0, *[float(valor) for valor in x]] for x in X]
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


def resolver_gauss(M, v, tolerancia=1e-12):
    """Resolve M z = v por eliminação de Gauss com pivoteamento parcial."""
    n = len(M)
    if n == 0 or len(v) != n or any(len(linha) != n for linha in M):
        raise ValueError("sistema inválido: M deve ser quadrada e compatível com v")

    M = [list(linha) for linha in M]
    v = list(v)

    for coluna in range(n):
        pivo = max(range(coluna, n), key=lambda linha: abs(M[linha][coluna]))
        if abs(M[pivo][coluna]) < tolerancia:
            raise ValueError("sistema singular ou mal condicionado")
        if pivo != coluna:
            M[coluna], M[pivo] = M[pivo], M[coluna]
            v[coluna], v[pivo] = v[pivo], v[coluna]

        for linha in range(coluna + 1, n):
            fator = M[linha][coluna] / M[coluna][coluna]
            for c in range(coluna, n):
                M[linha][c] -= fator * M[coluna][c]
            v[linha] -= fator * v[coluna]

    z = [0.0] * n
    for linha in reversed(range(n)):
        soma = sum(M[linha][c] * z[c] for c in range(linha + 1, n))
        z[linha] = (v[linha] - soma) / M[linha][linha]
    return z


def treinar_minimos_quadrados(X, y):
    """Resolve AᵀA z = Aᵀb por Gauss e devolve (w, b).

    A solução z = [b, w1, w2, ...] minimiza o erro quadrático entre A z e y.
    """
    A, b = montar_sistema(X, y)
    At = transposta(A)
    AtA = multiplicar(At, A)
    Atb = multiplicar_matriz_vetor(At, b)
    z = resolver_gauss(AtA, Atb)
    return z[1:], z[0]
