"""CLI do classificador linear por mínimos quadrados.

Sem argumentos, garante os CSVs em ``dados/`` (banknote, wdbc e os dois
casos), roda os dois modelos nos dois datasets e gera os gráficos e o resumo
em ``saida/``. Com argumentos, executa um pipeline específico.
"""

import argparse
import csv
import os
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, replace

from classificador.avaliacao import avaliar_modelo, matriz_confusao
from classificador.dados import ler_dados
from classificador.grafico import (
    caminho_preto_e_branco,
    equacao,
    plotar,
    plotar_comparacao,
    plotar_curva_limiar,
    plotar_histograma_escores,
    plotar_matriz_confusao,
)
from classificador.minimos_quadrados import treinar_minimos_quadrados
from classificador.modelo import classificar, funcao_linear
from classificador.normalizacao import ajustar_zscore, voltar_escala
from classificador.treinamento import treinar

URL_BANKNOTE = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "00267/data_banknote_authentication.txt"
)
URL_WDBC = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "breast-cancer-wisconsin/wdbc.data"
)

COLUNAS_WDBC = [
    "radius_mean",
    "texture_mean",
    "perimeter_mean",
    "area_mean",
    "smoothness_mean",
    "compactness_mean",
    "concavity_mean",
    "concave_points_mean",
    "symmetry_mean",
    "fractal_dimension_mean",
    "radius_se",
    "texture_se",
    "perimeter_se",
    "area_se",
    "smoothness_se",
    "compactness_se",
    "concavity_se",
    "concave_points_se",
    "symmetry_se",
    "fractal_dimension_se",
    "radius_worst",
    "texture_worst",
    "perimeter_worst",
    "area_worst",
    "smoothness_worst",
    "compactness_worst",
    "concavity_worst",
    "concave_points_worst",
    "symmetry_worst",
    "fractal_dimension_worst",
]

CONJUNTOS_PADRAO = [
    ("dados/banknote.csv", ["variance", "skewness"]),
    ("dados/wdbc.csv", ["radius_mean", "texture_mean"]),
]

TITULOS_PADRAO = {
    "banknote": "Notas falsas (+1) vs genuínas (-1)",
    "wdbc": "Câncer maligno (+1) vs benigno (-1)",
    "caso_1_separavel": "Caso 1 — separável",
    "caso_2_degenerado": "Caso 2 — não separável",
}


@dataclass
class Configuracao:
    csv: str
    colunas_x: list
    coluna_y: str = "classe"
    classe_positiva: str | None = None
    modelo: str = "mq"
    normalizar: bool = False
    taxa_aprendizado: float = 0.01
    epocas: int = 100
    semente: int = 42
    pontos: list | None = None
    sem_graficos: bool = False
    dir_saida: str = "graficos"
    titulo: str | None = None
    tamanho_ponto: float | None = None


def percentual(valor):
    """Formata uma fração 0–1 como porcentagem com vírgula decimal."""
    return f"{valor * 100:.1f}%".replace(".", ",")


def tamanho_ponto_automatico(quantidade):
    """Escolhe um tamanho de marcador legível para o número de pontos."""
    return max(6, min(45, round(4500 / quantidade)))


def rotulo_modelo(modelo):
    """Nome legível do modelo."""
    return "mínimos quadrados" if modelo == "mq" else "perceptron"


def criar_parser():
    """Monta o parser de argumentos da CLI."""
    parser = argparse.ArgumentParser(
        description=(
            "Classificador linear binário por mínimos quadrados ou perceptron. "
            "Sem argumentos, roda tudo (dados + dois modelos + gráficos)."
        )
    )
    parser.add_argument(
        "--csv",
        default="dados/banknote.csv",
        help="CSV de entrada (padrão: dados/banknote.csv)",
    )
    parser.add_argument(
        "--colunas-x",
        nargs=2,
        default=["variance", "skewness"],
        metavar=("X1", "X2"),
        help="nomes das duas colunas de features",
    )
    parser.add_argument(
        "--coluna-y",
        default="classe",
        help="nome da coluna de rótulos",
    )
    parser.add_argument(
        "--classe-positiva",
        default=None,
        help="valor do CSV que representa a classe +1 (obrigatório fora de ±1)",
    )
    parser.add_argument(
        "--modelo",
        choices=["mq", "perceptron"],
        default="mq",
        help="modelo de treinamento",
    )
    parser.add_argument(
        "--normalizar",
        action="store_true",
        help="aplica z-score nas features e devolve a reta à escala original",
    )
    parser.add_argument(
        "--taxa-aprendizado",
        type=float,
        default=0.01,
        help="taxa de aprendizado do perceptron",
    )
    parser.add_argument(
        "--epocas",
        type=int,
        default=100,
        help="número máximo de épocas do perceptron",
    )
    parser.add_argument(
        "--semente",
        type=int,
        default=42,
        help="semente do embaralhamento por época do perceptron",
    )
    parser.add_argument(
        "--dir-saida",
        default="graficos",
        help="pasta-base dos gráficos (padrão: graficos)",
    )
    parser.add_argument(
        "--titulo",
        default=None,
        help="título dos gráficos",
    )
    parser.add_argument(
        "--ponto",
        nargs=2,
        type=float,
        action="append",
        metavar=("X1", "X2"),
        help="classifica um ponto novo (pode repetir)",
    )
    parser.add_argument(
        "--tamanho-ponto",
        type=float,
        default=None,
        help="tamanho dos marcadores (padrão: automático)",
    )
    parser.add_argument(
        "--sem-graficos",
        action="store_true",
        help="imprime apenas as métricas",
    )
    return parser


def configuracao_de_args(args):
    """Converte os argumentos da CLI em uma Configuracao."""
    return Configuracao(
        csv=args.csv,
        colunas_x=args.colunas_x,
        coluna_y=args.coluna_y,
        classe_positiva=args.classe_positiva,
        modelo=args.modelo,
        normalizar=args.normalizar,
        taxa_aprendizado=args.taxa_aprendizado,
        epocas=args.epocas,
        semente=args.semente,
        pontos=args.ponto,
        sem_graficos=args.sem_graficos,
        dir_saida=args.dir_saida,
        titulo=args.titulo,
        tamanho_ponto=args.tamanho_ponto,
    )


def _baixar_banknote(destino):
    """Baixa o dataset UCI 267 e grava o CSV com rótulos -1/+1."""
    temporario = f"{destino}.tmp"
    temporario_csv = f"{destino}.novo"
    escritos = 0
    try:
        with urllib.request.urlopen(URL_BANKNOTE, timeout=30) as resposta:
            conteudo = resposta.read().decode("utf-8")

        with open(temporario, "w", newline="", encoding="utf-8") as arquivo:
            arquivo.write(conteudo)

        with (
            open(temporario, newline="", encoding="utf-8") as origem,
            open(temporario_csv, "w", newline="", encoding="utf-8") as saida,
        ):
            escritor = csv.writer(saida)
            escritor.writerow(["variance", "skewness", "curtosis", "entropy", "classe"])
            for linha in origem:
                partes = linha.strip().split(",")
                if len(partes) != 5:
                    continue
                classe = 1 if partes[4] == "1" else -1
                escritor.writerow([partes[0], partes[1], partes[2], partes[3], classe])
                escritos += 1

        if escritos == 0:
            raise RuntimeError("o dataset banknote baixado está vazio")

        os.replace(temporario_csv, destino)
    except (urllib.error.URLError, OSError) as erro:
        raise RuntimeError(f"falha ao baixar o dataset banknote: {erro}") from erro
    finally:
        for caminho in (temporario, temporario_csv):
            if os.path.exists(caminho):
                os.remove(caminho)


def _baixar_wdbc(destino):
    """Baixa o UCI 17 e grava as 30 features + classe (M=+1, B=-1)."""
    temporario = f"{destino}.novo"
    escritos = 0
    try:
        with urllib.request.urlopen(URL_WDBC, timeout=30) as resposta:
            texto = resposta.read().decode("utf-8-sig")

        with open(temporario, "w", newline="", encoding="utf-8") as saida:
            escritor = csv.writer(saida)
            escritor.writerow([*COLUNAS_WDBC, "classe"])
            for linha in texto.splitlines():
                partes = linha.strip().split(",")
                if len(partes) != 32:
                    continue
                classe = 1 if partes[1].strip() == "M" else -1
                escritor.writerow([*partes[2:], classe])
                escritos += 1

        if escritos == 0:
            raise RuntimeError("o dataset wdbc baixado está vazio")

        os.replace(temporario, destino)
    except (urllib.error.URLError, OSError, ValueError) as erro:
        raise RuntimeError(f"falha ao baixar o dataset wdbc: {erro}") from erro
    finally:
        if os.path.exists(temporario):
            os.remove(temporario)


def _gerar_caso_1(destino):
    """Cria o caso linearmente separável."""
    with open(destino, "w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.writer(arquivo)
        escritor.writerow(["x1", "x2", "classe"])
        for x1 in [1.0, 1.5, 2.0, 2.5, 3.0]:
            for x2 in [1.0, 1.5, 2.0, 2.5]:
                escritor.writerow([x1, x2, 1])
        for x1 in [-3.0, -2.5, -2.0, -1.5, -1.0]:
            for x2 in [-2.5, -2.0, -1.5, -1.0]:
                escritor.writerow([x1, x2, -1])


def _gerar_caso_2(destino):
    """Cria o caso não separável com rótulos ruidosos."""
    with open(destino, "w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.writer(arquivo)
        escritor.writerow(["x1", "x2", "classe"])
        for i in range(-5, 6):
            for j in range(-5, 6):
                if i == 0 and j == 0:
                    continue
                base = 1 if (i + j) > 0 else -1
                ruido = ((i * 7 + j * 3) % 10) < 5
                escritor.writerow([i, j, -base if ruido else base])


def garantir_dados():
    """Garante os CSVs dos 2 datasets e dos 2 casos em dados/."""
    os.makedirs("dados", exist_ok=True)

    banknote = os.path.join("dados", "banknote.csv")
    if not os.path.exists(banknote):
        _baixar_banknote(banknote)

    wdbc = os.path.join("dados", "wdbc.csv")
    if not os.path.exists(wdbc):
        _baixar_wdbc(wdbc)

    caso_1 = os.path.join("dados", "caso_1_separavel.csv")
    if not os.path.exists(caso_1):
        _gerar_caso_1(caso_1)

    caso_2 = os.path.join("dados", "caso_2_degenerado.csv")
    if not os.path.exists(caso_2):
        _gerar_caso_2(caso_2)


def executar_pipeline(config):
    """Executa leitura, treino, avaliação e gráficos de uma configuração."""
    rotulo = rotulo_modelo(config.modelo)
    print(f"\n=== {rotulo.upper()} | {config.csv} ===")

    try:
        X, y = ler_dados(
            config.csv,
            config.colunas_x,
            config.coluna_y,
            classe_positiva=config.classe_positiva,
        )
    except (OSError, ValueError) as erro:
        print(f"erro ao ler os dados: {erro}")
        return None

    X_treino = X
    medias = desvios = None
    if config.normalizar:
        try:
            X_treino, medias, desvios = ajustar_zscore(X)
        except ValueError as erro:
            print(f"erro na normalização: {erro}")
            return None

    try:
        if config.modelo == "mq":
            resultado = treinar_minimos_quadrados(X_treino, y)
            w_treino, b_treino = resultado.w, resultado.b
        else:
            resultado = treinar(
                X_treino,
                y,
                taxa_aprendizado=config.taxa_aprendizado,
                epocas=config.epocas,
                embaralhar=True,
                semente=config.semente,
            )
            w_treino, b_treino = resultado.w, resultado.b
    except ValueError as erro:
        print(f"erro ao treinar o modelo: {erro}")
        return None

    if config.normalizar:
        w, b = voltar_escala(w_treino, b_treino, medias, desvios)
    else:
        w, b = w_treino, b_treino

    previsoes = [classificar(ponto, w, b) for ponto in X]
    resumo = avaliar_modelo(X, y, w, b)
    contagens = matriz_confusao(y, previsoes)
    total = len(y)
    baseline = max(y.count(1), y.count(-1)) / total
    ganho = resumo["acuracia"] - baseline
    positivos = contagens["VP"] + contagens["FN"]
    negativos = contagens["VN"] + contagens["FP"]
    sensibilidade = contagens["VP"] / positivos if positivos else 0.0
    especificidade = contagens["VN"] / negativos if negativos else 0.0

    try:
        texto_equacao = equacao(w, b)
    except ValueError:
        texto_equacao = "reta indefinida (w = 0)"

    print(f"Pontos: {total} | +1: {y.count(1)} | -1: {y.count(-1)}")
    if config.normalizar:
        medias_txt = ", ".join(f"{valor:.4f}" for valor in medias)
        desvios_txt = ", ".join(f"{valor:.4f}" for valor in desvios)
        print(
            f"Normalização: z-score | médias = ({medias_txt}) | "
            f"desvios = ({desvios_txt})"
        )
    print(f"Baseline (maioria): {percentual(baseline)}")
    print(f"w = ({w[0]:.6f}, {w[1]:.6f}) | b = {b:.6f}")
    print(f"Reta: {texto_equacao}")
    ganho_formatado = f"{ganho * 100:+.1f}pp".replace(".", ",")
    print(
        f"Acurácia: {percentual(resumo['acuracia'])} "
        f"({resumo['acertos']}/{total}) | ganho: {ganho_formatado}"
    )
    print(
        f"VP={contagens['VP']} VN={contagens['VN']} "
        f"FP={contagens['FP']} FN={contagens['FN']}"
    )
    print(
        f"Sensibilidade: {percentual(sensibilidade)} | "
        f"Especificidade: {percentual(especificidade)}"
    )

    if config.modelo == "mq":
        print(f"det(AᵀA) = {resultado.determinante:.6e}")
        print(f"Resíduo ||Az-b||² = {resultado.residuo:.6f}")
        if config.normalizar:
            print("  (determinante e resíduo no espaço normalizado)")
    else:
        convergiu = "sim" if resultado.convergiu else "não"
        print(f"Convergiu: {convergiu} | épocas: {resultado.epocas}")
        if resultado.erros_por_epoca:
            print(
                f"Erros por época: {resultado.erros_por_epoca[0]} -> "
                f"{resultado.erros_por_epoca[-1]}"
            )

    if config.pontos:
        print("\nPontos novos:")
        for ponto in config.pontos:
            escore = funcao_linear(ponto, w, b)
            classe = classificar(ponto, w, b)
            print(
                f"  ({ponto[0]:g}, {ponto[1]:g}) -> "
                f"f(p) = {escore:.4f} -> Classe {classe:+d}"
            )

    nome_base = os.path.splitext(os.path.basename(config.csv))[0]
    resultado_pipeline = {
        "dataset": nome_base,
        "modelo": config.modelo,
        "n": total,
        "acuracia": resumo["acuracia"],
        "baseline": baseline,
        "ganho_pp": ganho * 100,
    }

    if config.sem_graficos:
        return resultado_pipeline

    dir_modelo = os.path.join(config.dir_saida, nome_base, config.modelo)
    titulo = config.titulo or TITULOS_PADRAO.get(
        nome_base, f"Classificador linear — {nome_base.replace('_', ' ')}"
    )
    titulo = f"{titulo} — {rotulo}"
    tamanho = config.tamanho_ponto or tamanho_ponto_automatico(total)

    try:
        caminhos = [
            plotar(
                X,
                y,
                w,
                b,
                os.path.join(dir_modelo, "dispersao.png"),
                titulo=titulo,
                tamanho_ponto=tamanho,
                rotulos_eixos=config.colunas_x,
            ),
            plotar(
                X,
                y,
                w,
                b,
                os.path.join(dir_modelo, "erros.png"),
                titulo=f"{titulo} — erros destacados",
                tamanho_ponto=tamanho,
                destacar_erros=True,
                rotulos_eixos=config.colunas_x,
            ),
            plotar_histograma_escores(
                X,
                y,
                w,
                b,
                os.path.join(dir_modelo, "histograma.png"),
                titulo=f"Distribuição dos escores — {titulo}",
            ),
            plotar_matriz_confusao(
                y,
                previsoes,
                os.path.join(dir_modelo, "confusao.png"),
                titulo=f"Matriz de confusão — {titulo}",
            ),
            plotar_curva_limiar(
                X,
                y,
                w,
                b,
                os.path.join(dir_modelo, "curva_limiar.png"),
                titulo=f"Métricas × limiar — {titulo}",
            ),
        ]
    except ValueError as erro:
        print(f"erro ao gerar os gráficos: {erro}")
        return None

    print("\nGráficos gerados:")
    for caminho in caminhos:
        print(f"  {caminho}")
        print(f"  {caminho_preto_e_branco(caminho)}")

    return resultado_pipeline


def escrever_resultados_csv(resultados, caminho="saida/resultados.csv"):
    """Escreve uma linha por dataset/modelo no CSV de resultados."""
    os.makedirs(os.path.dirname(caminho) or ".", exist_ok=True)
    with open(caminho, "w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.writer(arquivo)
        escritor.writerow(
            ["dataset", "modelo", "n", "acuracia", "baseline", "ganho_pp"]
        )
        for resultado in resultados:
            escritor.writerow(
                [
                    resultado["dataset"],
                    resultado["modelo"],
                    resultado["n"],
                    f"{resultado['acuracia']:.6f}",
                    f"{resultado['baseline']:.6f}",
                    f"{resultado['ganho_pp']:.2f}",
                ]
            )
    return caminho


def main():
    """Executa o modo padrão (tudo) ou um pipeline específico."""
    parser = criar_parser()
    args = parser.parse_args()
    modo_tudo = len(sys.argv) == 1

    if modo_tudo:
        try:
            garantir_dados()
        except (RuntimeError, OSError) as erro:
            print(f"erro ao preparar os dados: {erro}")
            return 1

        print("Dados prontos em dados/")
        base = configuracao_de_args(args)
        resultados = []
        for csv_path, colunas in CONJUNTOS_PADRAO:
            for modelo in ("mq", "perceptron"):
                config = replace(
                    base,
                    csv=csv_path,
                    colunas_x=colunas,
                    modelo=modelo,
                )
                resultado = executar_pipeline(config)
                if resultado is None:
                    return 1
                resultados.append(resultado)

        try:
            escrever_resultados_csv(resultados)
        except OSError as erro:
            print(f"erro ao escrever o resumo: {erro}")
            return 1

        try:
            comparacao = plotar_comparacao(
                resultados, os.path.join(args.dir_saida, "comparacao.png")
            )
        except ValueError as erro:
            print(f"erro ao gerar o gráfico comparativo: {erro}")
            return 1

        print("\nResumo salvo em saida/resultados.csv")
        print(f"Gráfico comparativo salvo em {comparacao}")
        return 0

    resultado = executar_pipeline(configuracao_de_args(args))
    return 0 if resultado is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
