import csv


def mapear_rotulos(rotulos, classe_positiva=None, mapa=None):
    """Converte os rótulos para +1/-1 sem depender da ordem das linhas.

    Se ``mapa`` for informado, ele é aplicado diretamente. Se
    ``classe_positiva`` for informada, essa classe vira +1 e a outra -1.
    Rótulos já numéricos em {-1, +1} são mantidos. Qualquer outro caso exige
    que a classe positiva seja declarada explicitamente.
    """
    if mapa is not None and classe_positiva is not None:
        raise ValueError("informe apenas 'mapa' ou 'classe_positiva', não os dois")

    if mapa is not None:
        return [mapa[rotulo] for rotulo in rotulos]

    classes = list(dict.fromkeys(rotulos))
    if classe_positiva is not None:
        if len(classes) != 2:
            raise ValueError(
                f"esperadas exatamente 2 classes na coluna de rótulo, encontradas {len(classes)}: {classes}"
            )
        if classe_positiva not in classes:
            raise ValueError(
                f"classe positiva {classe_positiva!r} não encontrada; classes: {classes}"
            )
        return [1 if rotulo == classe_positiva else -1 for rotulo in rotulos]

    try:
        numericos = [float(rotulo) for rotulo in rotulos]
    except (TypeError, ValueError):
        numericos = None

    if numericos is not None and set(numericos) <= {-1.0, 1.0}:
        return [int(valor) for valor in numericos]

    raise ValueError(
        "rótulos fora de {-1, +1}; informe a classe positiva (--classe-positiva) "
        "ou passe um mapa explícito"
    )


def ler_dados(
    caminho,
    colunas_x,
    coluna_y,
    classe_positiva=None,
    mapa_rotulos=None,
    delimitador=",",
):
    """Lê um CSV e devolve (X, y) pronto para o classificador.

    ``X`` é uma lista de pontos (listas de floats, uma por coluna em
    ``colunas_x``) e ``y`` é a lista de rótulos em +1/-1. Use
    ``classe_positiva`` para declarar qual valor do CSV representa a classe
    +1 quando os rótulos não forem -1/+1.
    """
    if not colunas_x:
        raise ValueError("informe pelo menos uma coluna de feature")
    if len(set(colunas_x)) != len(colunas_x):
        raise ValueError("as colunas de features não podem se repetir")
    if coluna_y in colunas_x:
        raise ValueError("a coluna de rótulo não pode ser usada como feature")

    with open(caminho, newline="", encoding="utf-8-sig") as arquivo:
        leitor = csv.DictReader(arquivo, delimiter=delimitador)
        if leitor.fieldnames is not None:
            leitor.fieldnames = [nome.strip() for nome in leitor.fieldnames]

        cabecalho = leitor.fieldnames or []
        faltando = [nome for nome in [*colunas_x, coluna_y] if nome not in cabecalho]
        if faltando:
            raise ValueError(
                f"colunas não encontradas no CSV: {faltando}; disponíveis: {cabecalho}"
            )

        X = []
        rotulos = []
        for numero_linha, linha in enumerate(leitor, start=2):
            if not linha or all(valor in (None, "") for valor in linha.values()):
                continue

            try:
                ponto = [float(linha[coluna].strip()) for coluna in colunas_x]
            except (AttributeError, TypeError, ValueError):
                raise ValueError(
                    f"valor não numérico nas colunas de features na linha {numero_linha}"
                )

            rotulo = linha[coluna_y]
            if rotulo is None or rotulo.strip() == "":
                raise ValueError(f"rótulo vazio na linha {numero_linha}")

            X.append(ponto)
            rotulos.append(rotulo.strip())

    if not X:
        raise ValueError(f"nenhum dado válido encontrado em {caminho}")

    y = mapear_rotulos(rotulos, classe_positiva=classe_positiva, mapa=mapa_rotulos)
    return X, y
