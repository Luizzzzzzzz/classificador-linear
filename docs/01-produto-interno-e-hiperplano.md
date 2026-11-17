# Produto interno e reta separadora

Este é o conceito base do Tema 2. A ideia é simples: damos um "empurrão" em
uma direção e medimos quanto um ponto está nessa direção.

## Produto interno

Se $w = (w_1, w_2)$ e $x = (x_1, x_2)$, o produto interno é:

$$
\langle w, x \rangle = w^T x = w_1 x_1 + w_2 x_2
$$

Ele responde: "quanto $x$ aponta para o mesmo lado de $w$?".

- Se o resultado é grande e positivo, $x$ está bem alinhado com $w$.
- Se é negativo, $x$ aponta para o lado oposto.
- Se é zero, os dois são perpendiculares (ângulo de $90^\circ$).

No código, isso é a função `funcao_linear` em `classificador/modelo.py`.

## A reta de separação

Uma reta no plano pode ser escrita assim:

$$
w_1 x + w_2 y + b = 0
$$

O vetor $w = (w_1, w_2)$ é **normal** à reta, ou seja, ele é perpendicular a
ela. O número $b$ só desloca a reta para um lado ou outro.

Se pegarmos qualquer ponto $p = (x, y)$ e calcularmos

$$
f(p) = w^T p + b = w_1 x + w_2 y + b,
$$

o sinal de $f(p)$ nos diz de qual lado da reta o ponto está:

- $f(p) > 0$: ponto de um lado;
- $f(p) < 0$: ponto do outro lado;
- $f(p) = 0$: ponto em cima da reta.

A distância assinada do ponto até a reta é:

$$
\frac{w^T p + b}{\|w\|}
$$

onde $\|w\| = \sqrt{w_1^2 + w_2^2}$.

## Exemplo do banknote

Depois do treino, o programa encontrou (números arredondados):

$$
-0{,}98x_1 - 0{,}20x_2 + 0{,}33 = 0
$$

Essa é a reta. O vetor normal é $(-0{,}98, -0{,}20)$: ele aponta para o lado
das cédulas falsas (classe $+1$). As genuínas (classe $-1$) ficam do outro lado.

## Por que isso é importante

O classificador não "olha" a reta. Ele só calcula $w^T p + b$ e olha o sinal.
A reta é apenas a fronteira geométrica onde esse número é zero. É por isso que
o produto interno é o coração do trabalho.
