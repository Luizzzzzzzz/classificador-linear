# O coração do classificador.
# Primeiro a gente calcula o escore f(x) = w·x + b.
# Depois decide a classe olhando só o sinal desse número.
# w é o vetor normal da reta; b só desloca a reta.


def funcao_linear(x, w, b):
    """Retorna f(x) = wᵀx + b, o produto interno de w e x somado ao viés b."""
    # cada peso multiplica a sua coordenada e a gente soma tudo
    return sum(wi * xi for wi, xi in zip(w, x)) + b


def classificar(x, w, b):
    """Classifica um ponto pelo escore: >= 0 chuta Classe 1, < 0 chuta Classe -1."""
    # o escore diz de qual lado da reta o ponto caiu
    escore = funcao_linear(x, w, b)
    # empate (zero) conta como classe +1
    return 1 if escore >= 0 else -1
