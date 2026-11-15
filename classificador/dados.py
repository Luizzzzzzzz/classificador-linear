import csv


def mapear_rotulos(rotulos, mapa=None):
    """Converte os rótulos para +1/-1."""
    if mapa is not None:
        return [mapa[rotulo] for rotulo in rotulos]

    try:
        numericos = [float(rotulo) for rotulo in rotulos]
    except (TypeError, ValueError):
        numericos = None

    if numericos is not None and set(numericos) <= {-1.0, 1.0}:
        return [int(valor) for valor in numericos]

    classes = list(dict.fromkeys(rotulos))
    if len(classes) != 2:
        raise ValueError(
            f"esperadas exatamente 2 classes na coluna de rótulo, encontradas {len(classes)}: {classes}"
        )

    conversao = {classes[0]: 1, classes[1]: -1}
    return [conversao[rotulo] for rotulo in rotulos]


def ler_dados(caminho, colunas_x, coluna_y, mapa_rotulos=None, delimitador=","):
    """Lê um CSV e devolve (X, y) pronto para o classificador."""
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

    y = mapear_rotulos(rotulos, mapa_rotulos)
    return X, y
