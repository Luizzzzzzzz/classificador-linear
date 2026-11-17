# Classificação e métricas

Depois do treino, temos $w$ e $b$. Agora precisamos medir se o modelo é bom.

## O chute

Para qualquer ponto novo $p = (x, y)$, calculamos o escore:

$$
s = f(p) = w^T p + b
$$

e chutamos:

$$
\hat{r} =
\begin{cases}
+1, & \text{se } s \geq 0 \\
-1, & \text{se } s < 0
\end{cases}
$$

Ou seja: se o ponto cair acima da reta, classe $+1$; se cair abaixo, classe
$-1$. Empate ($s = 0$) conta como $+1$.

## Matriz de confusão

Comparamos o chute $\hat{r}$ com o gabarito $r$:

| | chutou $+1$ | chutou $-1$ |
|:--|:--|:--|
| **é $+1$** | VP (acerto) | FN (erro) |
| **é $-1$** | FP (erro) | VN (acerto) |

- **VP** (verdadeiro positivo): era $+1$ e o modelo acertou.
- **VN** (verdadeiro negativo): era $-1$ e o modelo acertou.
- **FP** (falso positivo): era $-1$, mas o modelo chutou $+1$.
- **FN** (falso negativo): era $+1$, mas o modelo chutou $-1$.

## As métricas

$$
\text{Acurácia} = \frac{VP + VN}{n}
$$

$$
\text{Baseline} = \frac{\text{classe mais frequente}}{n}
$$

$$
\text{Ganho} = \text{Acurácia} - \text{Baseline}
$$

Também olhamos:

$$
\text{Sensibilidade} = \frac{VP}{VP + FN}
\qquad
\text{Especificidade} = \frac{VN}{VN + FP}
$$

A sensibilidade responde: "dos positivos reais, quantos o modelo achou?".
A especificidade responde: "dos negativos reais, quantos o modelo achou?".

## Números do projeto

| Dataset | Acurácia | Baseline | Ganho | VP | VN | FP | FN |
|:--|--:|--:|--:|--:|--:|--:|--:|
| banknote | 88,1% | 55,5% | +32,6pp | 515 | 694 | 68 | 95 |
| wdbc | 88,4% | 62,7% | +25,7pp | 158 | 345 | 12 | 54 |

No WDBC o positivo é **maligno**. A sensibilidade é $74{,}5\%$: o modelo
achou 158 dos 212 casos malignos e deixou passar 54 (falsos negativos). Num
problema de saúde, esse número importa mais que a acurácia — e é por isso que o
relatório precisa discutir o limiar.

## O zero não é sagrado

O corte padrão é $s = 0$. Mas nada impede de usar outro valor:

$$
\hat{r} = +1 \text{ se } s \geq \theta, \quad -1 \text{ se } s < \theta
$$

No banknote, o melhor limiar encontrado foi $\theta \approx -0{,}011$, com
acurácia de $88{,}2\%$ contra $88{,}1\%$ no limiar zero. A diferença é pequena,
mas mostra que o zero é uma escolha, não uma lei. O gráfico
`graficos/<dataset>/<modelo>/curva_limiar.png` mostra essas três curvas.
