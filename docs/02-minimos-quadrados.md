# Mínimos quadrados, passo a passo

Aqui está a conta que o programa faz para aprender a reta. Nada de biblioteca
pronta: montamos o sistema e resolvemos na mão.

## 1. Transformando pontos em matriz

Cada ponto de treino tem duas coordenadas e um rótulo: $p_i = (x_i, y_i)$ com
$r_i \in \{-1, +1\}$.

Para cada ponto, montamos uma linha com um `1` no início:

$$
a_i = [1,\ x_i,\ y_i]
$$

Juntando todas as linhas, temos a matriz $A$ (não confundir com a letra usada
para a coordenada):

$$
A = \begin{bmatrix}
1 & x_1 & y_1 \\
1 & x_2 & y_2 \\
\vdots & \vdots & \vdots \\
1 & x_n & y_n
\end{bmatrix},
\qquad
b = \begin{bmatrix} r_1 \\ r_2 \\ \vdots \\ r_n \end{bmatrix}
$$

O `1` da primeira coluna existe para que o termo independente $b$ apareça na
conta. Procuramos o vetor:

$$
z = \begin{bmatrix} b \\ w_1 \\ w_2 \end{bmatrix}
$$

## 2. O que estamos minimizando

Queremos que $Az$ fique o mais perto possível de $b$. O erro de cada linha é
$(Az - b)_i$. Somamos os quadrados:

$$
J(z) = \|Az - b\|^2
$$

Usar o quadrado é conveniente: erros para cima e para baixo não se cancelam, e
a conta tem uma fórmula limpa.

## 3. De onde vem a equação normal

Derivando $J(z)$ em relação a $z$ e igualando a zero:

$$
\nabla J(z) = 2A^T A z - 2A^T b = 0
$$

Logo:

$$
A^T A z = A^T b
$$

Isso é o **sistema normal**. Como temos 2 coordenadas mais o viés, $A^T A$ é
uma matriz $3 \times 3$ e $A^T b$ tem 3 posições.

## 4. Resolvendo com Gauss

O código usa eliminação de Gauss com **pivoteamento parcial**: em cada coluna,
escolhe a maior linha como pivô e troca de posição se preciso. Isso evita
dividir por números muito pequenos.

Tudo isso está em `classificador/minimos_quadrados.py`. Não usamos
`numpy.linalg.lstsq`, `sklearn` nem `scipy`.

## 5. O que o treino devolve

Além de $w$ e $b$, a função devolve dois números:

- $\det(A^T A)$: se estiver perto de zero, o sistema está mal condicionado (ou
  é singular). No banknote deu $6{,}678269 \times 10^{11}$; no WDBC,
  $3{,}775219 \times 10^{10}$.
- $\|Az - b\|^2$: a soma dos erros quadráticos. No banknote deu
  $549{,}690918$; no WDBC, $229{,}493476$.

## 6. Por que minimizar o erro não dá a maior margem

Essa é a pergunta que pode cair na banca.

O mínimos quadrados minimiza a distância entre $Az$ e os rótulos $\pm 1$. Ele
**não** pergunta qual reta deixa a maior folga entre as classes. Quem faz isso é
a SVM (máxima margem).

Consequência: o MQ pode errar um ponto mesmo quando existe uma reta que separa
as duas classes. Ele só quer acertar os valores $\pm 1$ no sentido do erro
quadrático.
