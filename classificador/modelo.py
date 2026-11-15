def funcao_linear(x, w, b):
    """Retorna f(x) = wᵀx + b, o produto interno de w e x somado ao viés b."""
    return sum(wi * xi for wi, xi in zip(w, x)) + b


def classificar(x, w, b):
    """Classifica um ponto pelo escore: >= 0 chuta Classe 1, < 0 chuta Classe -1."""
    escore = funcao_linear(x, w, b)
    return 1 if escore >= 0 else -1
