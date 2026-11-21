# Testes rápidos da lógica central, sem precisar de biblioteca extra.
# Cada teste guarda uma regra que não pode quebrar sem avisar.

import os
import tempfile
import unittest

from classificador.avaliacao import avaliar, matriz_confusao
from classificador.dados import ler_dados, mapear_rotulos
from classificador.minimos_quadrados import treinar_minimos_quadrados
from classificador.modelo import funcao_linear
from classificador.normalizacao import ajustar_zscore, voltar_escala


class TestRotulos(unittest.TestCase):
    # rótulos que já são -1/+1 têm que continuar iguais
    def test_pm1_preservado(self):
        self.assertEqual(mapear_rotulos(["1", "-1", "1"]), [1, -1, 1])

    def test_classe_positiva_explicita_nao_depende_da_ordem(self):
        self.assertEqual(mapear_rotulos(["B", "A"], classe_positiva="B"), [1, -1])
        self.assertEqual(mapear_rotulos(["A", "B"], classe_positiva="B"), [-1, 1])

    def test_rejeita_rotulos_fora_de_pm1(self):
        with self.assertRaises(ValueError):
            mapear_rotulos(["0", "1"])
        with self.assertRaises(ValueError):
            mapear_rotulos(["a", "b"], mapa={"a": 0, "b": 1})


class TestMinimosQuadrados(unittest.TestCase):
    # confere a conta passo a passo do exemplo dos 3 pontos
    def test_exemplo_tres_pontos(self):
        X = [[1.0, 1.0], [1.5, 2.0], [-1.0, -1.0]]
        y = [1, 1, -1]
        resultado = treinar_minimos_quadrados(X, y)

        self.assertAlmostEqual(resultado.w[0], 2.0, places=9)
        self.assertAlmostEqual(resultado.w[1], -1.0, places=9)
        self.assertAlmostEqual(resultado.b, 0.0, places=9)
        self.assertAlmostEqual(resultado.determinante, 1.0, places=9)
        self.assertAlmostEqual(resultado.residuo, 0.0, places=9)


class TestAvaliacao(unittest.TestCase):
    def test_matriz_confusao_rejeita_rotulo_invalido(self):
        with self.assertRaises(ValueError):
            matriz_confusao([1, 0], [1, -1])
        with self.assertRaises(ValueError):
            avaliar([1, 0], [1, -1])


class TestNormalizacao(unittest.TestCase):
    # normalizar e voltar não pode mudar o escore do ponto
    def test_ida_e_volta_preserva_escore(self):
        X = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        X_normalizado, medias, desvios = ajustar_zscore(X)
        w = [0.5, -0.25]
        b = 0.1

        w_original, b_original = voltar_escala(w, b, medias, desvios)
        for ponto, ponto_normalizado in zip(X, X_normalizado):
            escore_original = funcao_linear(ponto, w_original, b_original)
            escore_normalizado = funcao_linear(ponto_normalizado, w, b)
            self.assertAlmostEqual(escore_original, escore_normalizado, places=9)


class TestLeituraCSV(unittest.TestCase):
    # se faltar uma coluna, o erro deve avisar quais existem
    def test_coluna_faltando(self):
        with tempfile.TemporaryDirectory() as diretorio:
            caminho = os.path.join(diretorio, "teste.csv")
            with open(caminho, "w", newline="", encoding="utf-8") as arquivo:
                arquivo.write("x1,classe\n1,1\n2,-1\n")

            with self.assertRaises(ValueError) as contexto:
                ler_dados(caminho, ["x1", "x2"], "classe")
            self.assertIn("colunas não encontradas", str(contexto.exception))


if __name__ == "__main__":
    unittest.main()
