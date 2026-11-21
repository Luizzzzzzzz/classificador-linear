# Todos os desenhos do trabalho ficam aqui.
# A ideia é sempre a mesma: pontos, reta e regiões de decisão.
# Cada função salva o PNG colorido e também uma versão em preto e branco.

import math
import os
from contextlib import contextmanager

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, PercentFormatter
from PIL import Image

from .avaliacao import matriz_confusao, metricas_por_limiar
from .modelo import classificar, funcao_linear


@contextmanager
def _figuras_temporarias():
    """Fecha qualquer figura criada durante a chamada, mesmo se houver erro."""
    # lembra quais figuras já estavam abertas antes
    antes = set(plt.get_fignums())
    try:
        yield
    finally:
        # fecha só as que esta função criou, mesmo se tiver dado erro
        for numero in set(plt.get_fignums()) - antes:
            plt.close(numero)


def caminho_preto_e_branco(caminho):
    """Devolve o caminho da versão em preto e branco (sufixo _pb)."""
    raiz, extensao = os.path.splitext(caminho)
    return f"{raiz}_pb{extensao}"


def _salvar_versoes(figura, caminho, gerar_pb=True, **opcoes):
    """Salva o PNG colorido e, se pedido, uma versão em tons de cinza."""
    os.makedirs(os.path.dirname(caminho) or ".", exist_ok=True)
    # salva primeiro a versão colorida
    figura.savefig(caminho, dpi=150, **opcoes)
    if gerar_pb:
        # depois converte para tons de cinza (fica melhor para imprimir)
        destino = caminho_preto_e_branco(caminho)
        with Image.open(caminho) as imagem:
            cinza = imagem.convert("L")
            cinza = cinza.point(lambda p: int(255 * (p / 255) ** 1.5))
            cinza.save(destino)
    return caminho


def limites(X, margem=0.1):
    """Calcula (xmin, xmax, ymin, ymax) com folga em torno dos pontos."""
    if not X:
        raise ValueError("não há pontos para definir os limites")

    xs = [ponto[0] for ponto in X]
    ys = [ponto[1] for ponto in X]

    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)

    # deixa uma folga para os pontos não colarem na borda
    largura = (xmax - xmin) or 1.0
    altura = (ymax - ymin) or 1.0

    return (
        xmin - largura * margem,
        xmax + largura * margem,
        ymin - altura * margem,
        ymax + altura * margem,
    )


def gerar_grade(X, resolucao=200, margem=0.1):
    """Gera os eixos xs e ys de uma malha que cobre a área dos pontos."""
    if resolucao < 2:
        raise ValueError("a resolução da grade deve ser pelo menos 2")

    # malha retangular que cobre toda a área dos pontos
    xmin, xmax, ymin, ymax = limites(X, margem)
    passo_x = (xmax - xmin) / (resolucao - 1)
    passo_y = (ymax - ymin) / (resolucao - 1)

    xs = [xmin + i * passo_x for i in range(resolucao)]
    ys = [ymin + j * passo_y for j in range(resolucao)]
    return xs, ys


def classificar_grade(xs, ys, w, b):
    """Classifica todos os pontos da malha, devolvendo uma matriz Z."""
    return [[classificar([x, y], w, b) for x in xs] for y in ys]


def reta_decisao(w, b, xmin, xmax, ymin, ymax):
    """Devolve dois pontos da reta w₁x + w₂y + b = 0, inclusive se vertical."""
    if len(w) != 2:
        raise ValueError("o gráfico bidimensional exige exatamente 2 pesos")

    tolerancia = 1e-12
    # caso normal: y = -(w1*x + b)/w2
    if abs(w[1]) > tolerancia:
        xs = [xmin, xmax]
        ys = [-(w[0] * x + b) / w[1] for x in xs]
    # se w2 é quase zero, a reta fica vertical
    elif abs(w[0]) > tolerancia:
        x = -b / w[0]
        xs = [x, x]
        ys = [ymin, ymax]
    else:
        raise ValueError("w nulo não define uma reta de decisão")

    return xs, ys


def normalizar(w, b):
    """Divide w e b pela norma de w para obter um vetor normal unitário."""
    norma = math.sqrt(sum(wi * wi for wi in w))
    if norma < 1e-12:
        raise ValueError("não é possível normalizar um vetor nulo")
    return [wi / norma for wi in w], b / norma


def equacao(w, b):
    """Formata a equação da reta com o vetor normal unitário."""
    if len(w) != 2:
        raise ValueError("a equação bidimensional exige exatamente 2 pesos")

    # deixa o vetor normal com tamanho 1 para a conta ficar legível
    w_unit, b_unit = normalizar(w, b)
    return f"{w_unit[0]:+.2f}x₁ {w_unit[1]:+.2f}x₂ {b_unit:+.2f} = 0"


@_figuras_temporarias()
def plotar(
    X,
    y,
    w,
    b,
    caminho="graficos/grafico.png",
    titulo="Classificador linear",
    mostrar_regioes=True,
    resolucao=200,
    mostrar=False,
    margem=0.1,
    tamanho_ponto=45,
    destacar_erros=False,
    gerar_pb=True,
    rotulos_eixos=None,
):
    """Desenha pontos, reta e regiões de decisão; salva o PNG e devolve o caminho."""
    if len(X) != len(y):
        raise ValueError(f"tamanhos diferentes: {len(X)} pontos e {len(y)} rótulos")
    if len(w) != 2:
        raise ValueError("o gráfico bidimensional exige exatamente 2 pesos")
    if rotulos_eixos is not None and len(rotulos_eixos) != 2:
        raise ValueError("rotulos_eixos deve ter exatamente 2 nomes")

    xmin, xmax, ymin, ymax = limites(X, margem)
    figura, eixo = plt.subplots(figsize=(9, 6))

    if mostrar_regioes:
        # pinta o fundo: de um lado a classe +1, do outro a classe -1
        xs, ys = gerar_grade(X, resolucao, margem)
        Z = classificar_grade(xs, ys, w, b)
        eixo.contourf(
            xs,
            ys,
            Z,
            levels=[-1.5, 0, 1.5],
            colors=["#e39a9a", "#c8dcf5"],
            alpha=0.55,
            hatches=["", "//"],
        )

    # cada classe tem cor e formato diferentes (ajuda no preto e branco)
    cores = {1: "#1f77b4", -1: "#d62728"}
    marcadores = {1: "o", -1: "^"}
    for classe, rotulo in ((1, "Classe +1"), (-1, "Classe -1")):
        pontos = [ponto for ponto, valor in zip(X, y) if valor == classe]
        if pontos:
            eixo.scatter(
                [ponto[0] for ponto in pontos],
                [ponto[1] for ponto in pontos],
                c=cores[classe],
                marker=marcadores[classe],
                edgecolors="black",
                linewidths=0.4,
                s=tamanho_ponto,
                label=rotulo,
                zorder=3,
            )

    if destacar_erros:
        # marca com símbolo próprio quem o modelo errou
        previsoes = [classificar(ponto, w, b) for ponto in X]
        falsos_positivos = [
            ponto
            for ponto, real, chute in zip(X, y, previsoes)
            if real == -1 and chute == 1
        ]
        falsos_negativos = [
            ponto
            for ponto, real, chute in zip(X, y, previsoes)
            if real == 1 and chute == -1
        ]

        if falsos_positivos:
            eixo.scatter(
                [ponto[0] for ponto in falsos_positivos],
                [ponto[1] for ponto in falsos_positivos],
                marker="X",
                c="#ff7f0e",
                edgecolors="black",
                linewidths=0.6,
                s=tamanho_ponto * 2.2,
                label=f"Falso positivo ({len(falsos_positivos)})",
                zorder=5,
            )
        if falsos_negativos:
            eixo.scatter(
                [ponto[0] for ponto in falsos_negativos],
                [ponto[1] for ponto in falsos_negativos],
                marker="s",
                c="#9467bd",
                edgecolors="black",
                linewidths=0.6,
                s=tamanho_ponto * 2.0,
                label=f"Falso negativo ({len(falsos_negativos)})",
                zorder=5,
            )

    # desenha a reta separadora por cima de tudo
    xs_reta, ys_reta = reta_decisao(w, b, xmin, xmax, ymin, ymax)
    eixo.plot(
        xs_reta,
        ys_reta,
        color="black",
        linewidth=2,
        label=f"Reta: {equacao(w, b)}",
        zorder=4,
    )

    eixo.set_xlim(xmin, xmax)
    eixo.set_ylim(ymin, ymax)
    # usa os nomes reais das colunas quando o main passa esses rótulos
    if rotulos_eixos:
        eixo.set_xlabel(rotulos_eixos[0])
        eixo.set_ylabel(rotulos_eixos[1])
    else:
        eixo.set_xlabel("x₁")
        eixo.set_ylabel("x₂")
    eixo.set_title(titulo)
    eixo.grid(alpha=0.3)
    eixo.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
    figura.tight_layout()
    _salvar_versoes(figura, caminho, gerar_pb, bbox_inches="tight")

    if mostrar:
        plt.show()
    plt.close(figura)
    return caminho


@_figuras_temporarias()
def plotar_histograma_escores(
    X,
    y,
    w,
    b,
    caminho="graficos/histograma_escores.png",
    titulo="Distribuição dos escores f(p) = w·p + b",
    bins=30,
    mostrar=False,
    gerar_pb=True,
):
    """Sobrepõe os histogramas dos escores das duas classes e marca o zero."""
    if len(X) != len(y):
        raise ValueError(f"tamanhos diferentes: {len(X)} pontos e {len(y)} rótulos")
    if bins < 2:
        raise ValueError("o histograma precisa de pelo menos 2 bins")

    escores_positivos = [
        funcao_linear(ponto, w, b) for ponto, valor in zip(X, y) if valor == 1
    ]
    escores_negativos = [
        funcao_linear(ponto, w, b) for ponto, valor in zip(X, y) if valor == -1
    ]
    if not escores_positivos or not escores_negativos:
        raise ValueError("são necessárias as duas classes para o histograma")

    todos = escores_positivos + escores_negativos
    minimo, maximo = min(todos), max(todos)
    if minimo == maximo:
        minimo -= 0.5
        maximo += 0.5
    passo = (maximo - minimo) / bins
    bordas = [minimo + i * passo for i in range(bins + 1)]

    figura, eixo = plt.subplots(figsize=(8, 5))
    # dois histogramas alinhados para ver onde as classes se misturam
    eixo.hist(
        escores_negativos,
        bins=bordas,
        color="#d62728",
        alpha=0.6,
        edgecolor="black",
        linewidth=0.4,
        hatch="//",
        label="Classe -1",
    )
    eixo.hist(
        escores_positivos,
        bins=bordas,
        color="#1f77b4",
        alpha=0.6,
        edgecolor="black",
        linewidth=0.4,
        label="Classe +1",
    )
    # linha no zero: é o corte que o classificador usa
    eixo.axvline(
        0,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label="Fronteira (escore = 0)",
    )
    eixo.set_xlabel("f(p) = w·p + b")
    eixo.set_ylabel("Quantidade de pontos")
    eixo.set_title(titulo)
    eixo.grid(alpha=0.25)
    eixo.legend(loc="best")
    figura.tight_layout()
    os.makedirs(os.path.dirname(caminho) or ".", exist_ok=True)
    _salvar_versoes(figura, caminho, gerar_pb)

    if mostrar:
        plt.show()
    plt.close(figura)
    return caminho


@_figuras_temporarias()
def plotar_matriz_confusao(
    y_real,
    y_previsto,
    caminho="graficos/matriz_confusao.png",
    titulo="Matriz de confusão",
    mostrar=False,
    gerar_pb=True,
):
    """Desenha a matriz 2×2 com VP, VN, FP e FN e a acurácia no título."""
    if not y_real:
        raise ValueError("não há dados para a matriz de confusão")

    contagens = matriz_confusao(y_real, y_previsto)
    total = sum(contagens.values())
    acuracia = (contagens["VP"] + contagens["VN"]) / total
    matriz = [
        [contagens["VN"], contagens["FP"]],
        [contagens["FN"], contagens["VP"]],
    ]
    nomes = [["VN", "FP"], ["FN", "VP"]]

    figura, eixo = plt.subplots(figsize=(5.5, 4.8))
    # linha é o gabarito; coluna é o chute do modelo
    imagem = eixo.imshow(matriz, cmap="Blues", origin="upper")
    eixo.set_xticks([0, 1], labels=["Predita -1", "Predita +1"])
    eixo.set_yticks([0, 1], labels=["Real -1", "Real +1"])
    eixo.set_xlabel("Classe predita")
    eixo.set_ylabel("Classe real")

    maior = max(max(linha) for linha in matriz) or 1
    for linha in range(2):
        for coluna in range(2):
            valor = matriz[linha][coluna]
            cor = "white" if valor > 0.8 * maior else "black"
            contorno = "black" if cor == "white" else "white"
            eixo.text(
                coluna,
                linha,
                f"{nomes[linha][coluna]}\n{valor}",
                ha="center",
                va="center",
                color=cor,
                fontsize=14,
                fontweight="bold",
                path_effects=[pe.withStroke(linewidth=2.5, foreground=contorno)],
            )

    percentual = f"{acuracia * 100:.1f}".replace(".", ",")
    eixo.set_title(f"{titulo}\nAcurácia: {percentual}%")
    figura.colorbar(imagem, ax=eixo, label="Quantidade")
    figura.tight_layout()
    os.makedirs(os.path.dirname(caminho) or ".", exist_ok=True)
    _salvar_versoes(figura, caminho, gerar_pb)

    if mostrar:
        plt.show()
    plt.close(figura)
    return caminho


@_figuras_temporarias()
def plotar_curva_limiar(
    X,
    y,
    w,
    b,
    caminho="graficos/curva_limiar.png",
    titulo="Acurácia, sensibilidade e especificidade × limiar",
    passos=200,
    mostrar=False,
    gerar_pb=True,
):
    """Varre o limiar de decisão e plota acurácia, sensibilidade e especificidade."""
    if len(X) != len(y):
        raise ValueError(f"tamanhos diferentes: {len(X)} pontos e {len(y)} rótulos")
    if not X:
        raise ValueError("não há dados para a curva de limiar")
    if passos < 2:
        raise ValueError("a curva precisa de pelo menos 2 limiares")

    # varre vários cortes e mede o que acontece em cada um
    escores = [funcao_linear(ponto, w, b) for ponto in X]
    minimo, maximo = min(escores), max(escores)
    if minimo == maximo:
        minimo -= 0.5
        maximo += 0.5

    folga = (maximo - minimo) * 0.01
    inicio = minimo - folga
    fim = maximo + folga
    limiares = [inicio + i * (fim - inicio) / (passos - 1) for i in range(passos)]

    acuracias, sensibilidades, especificidades = [], [], []
    for limiar in limiares:
        metricas = metricas_por_limiar(y, escores, limiar)
        acuracias.append(metricas["acuracia"])
        sensibilidades.append(metricas["sensibilidade"])
        especificidades.append(metricas["especificidade"])

    indice_melhor = max(range(len(limiares)), key=lambda i: acuracias[i])
    melhor_limiar = limiares[indice_melhor]
    melhor_acuracia = acuracias[indice_melhor]
    limiar_formatado = f"{melhor_limiar:.3f}".replace(".", ",")

    figura, eixo = plt.subplots(figsize=(9, 5.5))
    eixo.plot(limiares, acuracias, color="black", linewidth=2.5, label="Acurácia")
    eixo.plot(
        limiares,
        sensibilidades,
        color="#1f77b4",
        linestyle="--",
        linewidth=1.8,
        label="Sensibilidade (VP/(VP+FN))",
    )
    eixo.plot(
        limiares,
        especificidades,
        color="#d62728",
        linestyle="-.",
        linewidth=1.8,
        label="Especificidade (VN/(VN+FP))",
    )
    eixo.axvline(
        0,
        color="gray",
        linestyle="--",
        linewidth=1.5,
        label="Limiar padrão (0)",
    )
    eixo.axvline(
        melhor_limiar,
        color="#2ca02c",
        linestyle=":",
        linewidth=1.5,
        label=f"Melhor acurácia (limiar = {limiar_formatado})",
    )
    eixo.plot(
        [melhor_limiar],
        [melhor_acuracia],
        marker="*",
        markersize=14,
        color="#2ca02c",
        markeredgecolor="black",
        linestyle="none",
        zorder=5,
    )
    eixo.set_xlabel("Limiar de decisão")
    eixo.set_ylabel("Métrica")
    eixo.set_ylim(-0.02, 1.02)
    eixo.xaxis.set_major_formatter(
        FuncFormatter(lambda valor, posicao: f"{valor:.3f}".replace(".", ","))
    )
    eixo.yaxis.set_major_formatter(PercentFormatter(1.0))
    eixo.set_title(titulo)
    eixo.grid(alpha=0.3)
    eixo.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
    figura.tight_layout()
    os.makedirs(os.path.dirname(caminho) or ".", exist_ok=True)
    _salvar_versoes(figura, caminho, gerar_pb, bbox_inches="tight")

    if mostrar:
        plt.show()
    plt.close(figura)
    return caminho


@_figuras_temporarias()
def plotar_comparacao(
    resultados,
    caminho="graficos/comparacao.png",
    titulo="Comparação de modelos — acurácia",
    gerar_pb=True,
    mostrar=False,
):
    """Barras agrupadas de acurácia por dataset e modelo."""
    if not resultados:
        raise ValueError("não há resultados para comparar")

    datasets = list(dict.fromkeys(item["dataset"] for item in resultados))
    modelos = list(dict.fromkeys(item["modelo"] for item in resultados))
    if len(modelos) != 2:
        raise ValueError("a comparação espera exatamente 2 modelos")

    acuracias = {
        (item["dataset"], item["modelo"]): item["acuracia"] for item in resultados
    }
    cores = {"mq": "#d9e4f2", "perceptron": "#d62728"}
    rotulos = {"mq": "MQ", "perceptron": "Perceptron"}
    hachuras = {"mq": "", "perceptron": "//"}

    largura = 0.35
    posicoes = list(range(len(datasets)))
    figura, eixo = plt.subplots(figsize=(8, 5))

    # uma barra para MQ e outra para perceptron em cada dataset
    for indice, modelo in enumerate(modelos):
        valores = [acuracias.get((dataset, modelo), 0.0) for dataset in datasets]
        deslocados = [posicao + (indice - 0.5) * largura for posicao in posicoes]
        barras = eixo.bar(
            deslocados,
            valores,
            largura,
            label=rotulos.get(modelo, modelo),
            color=cores.get(modelo, "#888888"),
            hatch=hachuras.get(modelo, ""),
            edgecolor="black",
            linewidth=0.6,
            zorder=3,
        )
        for barra, valor in zip(barras, valores):
            eixo.text(
                barra.get_x() + barra.get_width() / 2,
                valor + 0.012,
                f"{valor * 100:.1f}%".replace(".", ","),
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    eixo.set_xticks(posicoes, [nome.replace("_", " ") for nome in datasets])
    eixo.set_ylabel("Acurácia")
    eixo.set_ylim(0, 1.08)
    eixo.yaxis.set_major_formatter(PercentFormatter(1.0))
    eixo.set_title(titulo)
    eixo.grid(axis="y", alpha=0.3, zorder=0)
    eixo.legend(loc="upper right")
    figura.tight_layout()
    _salvar_versoes(figura, caminho, gerar_pb, bbox_inches="tight")

    if mostrar:
        plt.show()
    plt.close(figura)
    return caminho
