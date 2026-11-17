# Exemplo numérico completo

Vamos fazer a conta do começo ao fim com três pontos do `caso_1_separavel`.

## Os pontos

| $x$ | $y$ | rótulo $r$ |
|--:|--:|--:|
| 1,0 | 1,0 | +1 |
| 1,5 | 2,0 | +1 |
| -1,0 | -1,0 | -1 |

## 1. Montando $A$ e $b$

Cada linha ganha um `1` na frente:

$$
A = \begin{bmatrix}
1 & 1{,}0 & 1{,}0 \\
1 & 1{,}5 & 2{,}0 \\
1 & -1{,}0 & -1{,}0
\end{bmatrix},
\qquad
b = \begin{bmatrix} 1 \\ 1 \\ -1 \end{bmatrix}
$$

## 2. Calculando $A^T A$ e $A^T b$

$$
A^T A = \begin{bmatrix}
3 & 1{,}5 & 2 \\
1{,}5 & 4{,}25 & 5 \\
2 & 5 & 6
\end{bmatrix},
\qquad
A^T b = \begin{bmatrix} 1 \\ 3{,}5 \\ 4 \end{bmatrix}
$$

O determinante aqui é $\det(A^T A) = 1$: sistema bem comportado.

## 3. Resolvendo o sistema

Resolvendo $A^T A z = A^T b$:

$$
z = \begin{bmatrix} b \\ w_1 \\ w_2 \end{bmatrix}
= \begin{bmatrix} 0 \\ 2 \\ -1 \end{bmatrix}
$$

Ou seja: $b = 0$, $w_1 = 2$, $w_2 = -1$.

## 4. A reta e a classificação

A reta é:

$$
2x - y = 0
$$

Ou, na forma com o vetor normal unitário:

$$
0{,}89x_1 - 0{,}45x_2 = 0
$$

Testando um ponto novo $p = (2, 2)$:

$$
f(p) = 2 \cdot 2 + (-1)\cdot 2 + 0 = 2 > 0
$$

Logo, chutamos classe $+1$.

O resíduo nesse exemplo é praticamente zero ($\approx 10^{-29}$): os três
pontos caem exatamente onde o modelo espera. Isso é esperado porque o caso é
separável e pequeno.

## 5. Resultado do caso completo

Rodando o `caso_1_separavel.csv` inteiro (40 pontos), o programa imprime:

```text
w = (0.212766, 0.297872) | b = 0.000000
det(AᵀA) = 1.880000e+05
Resíduo ||Az-b||² = 2.127660
Acurácia: 100,0% (40/40)
```

A reta muda em relação ao exemplo de 3 pontos porque agora temos 40 pontos,
mas a ideia é a mesma.

## 6. Números finais dos datasets reais

| Dataset | $w_1$ | $w_2$ | $b$ | Acurácia |
|:--|--:|--:|--:|--:|
| banknote | -0,228358 | -0,046126 | 0,076930 | 88,1% |
| wdbc | 0,182730 | 0,044950 | -3,703388 | 88,4% |

Esses valores aparecem em `graficos/<dataset>/mq/dispersao.png` junto com a
equação da reta.
