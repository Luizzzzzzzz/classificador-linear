# Perceptron e normalização

## O perceptron

O perceptron é outro jeito de aprender $w$ e $b$. A regra é muito simples:

1. começa com $w = 0$ e $b = 0$;
2. para cada ponto, calcula o chute;
3. se acertou, **não faz nada**;
4. se errou, empurra os pesos na direção do rótulo correto.

A atualização é:

$$
w \leftarrow w + \eta\, r_i\, x_i
\qquad
b \leftarrow b + \eta\, r_i
$$

onde $\eta$ é a taxa de aprendizado. Se $r_i = +1$, somamos; se $r_i = -1$,
subtraímos.

O código embaralha a ordem dos pontos a cada época (com semente fixa) para que
a ordem do CSV não influencie o resultado. Se uma época termina sem erros, o
treino para e a flag `convergiu` vira `True`.

## Resultados do perceptron

| Dataset | Acurácia | Convergiu? |
|:--|--:|:--|
| caso separável | 100,0% | sim, em 2 épocas |
| banknote | 86,7% | não (100 épocas) |
| wdbc | 82,1% | não (100 épocas) |

Nos dados reais ele não converge porque as classes não são perfeitamente
separáveis por uma reta. Mesmo assim, o perceptron fica perto do MQ no
banknote; no WDBC a diferença é maior.

## Normalização z-score

Às vezes as features têm escalas muito diferentes. No banknote, `variance` tem
desvio ≈ $2{,}84$ e `skewness` ≈ $5{,}87$. O z-score coloca as duas na mesma
escala:

$$
x'_j = \frac{x_j - \mu_j}{\sigma_j}
$$

onde $\mu_j$ é a média e $\sigma_j$ o desvio padrão da feature $j$ no conjunto
de treino.

Depois de treinar com os dados normalizados, precisamos voltar à escala
original, senão a reta sai em unidades erradas:

$$
w_j^{\text{original}} = \frac{w_j}{\sigma_j}
$$

$$
b^{\text{original}} = b - \sum_j \frac{w_j \mu_j}{\sigma_j}
$$

A fórmula vem de substituir $x'_j = (x_j - \mu_j)/\sigma_j$ em
$w^T x' + b$ e reorganizar.

## Números do banknote normalizado

- médias: $(0{,}4337,\ 1{,}9224)$
- desvios: $(2{,}8417,\ 5{,}8669)$
- $\det(A^T A)$ no espaço normalizado: $2{,}402597 \times 10^{09}$
- resíduo: $549{,}690918$ (o mesmo de antes — a transformação é linear)

O MQ continua com $88{,}1\%$ de acurácia. O perceptron, com normalização, sobe
para $88{,}0\%$, mas ainda não converge em 100 épocas.
