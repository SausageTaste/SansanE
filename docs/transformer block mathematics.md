# Mathematics of `transformer_block`

This document follows [`transformer_block`](../src/main.cpp) in execution order. It describes **one GPT-2 Transformer layer** during inference. The input already contains token and position embeddings; the final LayerNorm and vocabulary projection happen after all Transformer blocks.

## Shapes and notation

Let $T$ be the number of tokens in the current context, $C$ the number of channels, $H$ the number of attention heads, and $d=C/H$ the channels per head. For GPT-2 124M, $C=768$, $H=12$, and $d=64$. Positions and channels below use zero-based indices, as in the code.

| Symbol | Shape | Meaning |
| --- | --- | --- |
| $X$ | $T\times C$ | Input residual stream (`input`) |
| $Q,K,V$ | each $T\times C$ | Query, key, and value projections |
| $A$ | $T\times C$ | Concatenated attention-head outputs |
| $R$ | $T\times C$ | Residual stream after attention |
| $Y$ | $T\times C$ | Block output |

Each row is one token position. A `linear` call with input $U\in\mathbb{R}^{T\times I}$, stored weight $W\in\mathbb{R}^{O\times I}$, and bias $b\in\mathbb{R}^{O}$ computes

$$
\operatorname{Linear}(U;W,b)_{t,o}
=b_o+\sum_{i=0}^{I-1}W_{o,i}U_{t,i},
\qquad\text{or}\qquad
\operatorname{Linear}(U;W,b)=UW^{\mathsf T}+b.
$$

The bias is added to every row. The transpose matters: checkpoint weights are stored with **output channels first**, while token activations are rows.

## 1. Normalize for attention

`layer_norm(input, ln1w, ln1b)` independently normalizes the channels of each token row. For position $t$,

$$
\mu_t=\frac{1}{C}\sum_{c=0}^{C-1}X_{t,c},
\qquad
\sigma_t^2=\frac{1}{C}\sum_{c=0}^{C-1}(X_{t,c}-\mu_t)^2,
$$

$$
N^{(1)}_{t,c}
=\gamma^{(1)}_c
\frac{X_{t,c}-\mu_t}{\sqrt{\sigma_t^2+\varepsilon}}
+\beta^{(1)}_c,
\qquad \varepsilon=10^{-5}.
$$

Here `ln1w` is $\gamma^{(1)}$ and `ln1b` is $\beta^{(1)}$. The variance divides by $C$, not $C-1$. $X$ is kept for the later residual addition: this is **pre-normalization**.

## 2. Project to queries, keys, and values

The block makes one linear projection with $3C$ output channels:

$$
P=\operatorname{Linear}(N^{(1)};W_{qkv},b_{qkv})
\in\mathbb{R}^{T\times 3C},
\qquad
W_{qkv}\in\mathbb{R}^{3C\times C}.
$$

`split_qkv(C)` takes consecutive channel ranges, without adding or changing values:

$$
Q_{t,c}=P_{t,c},\qquad
K_{t,c}=P_{t,C+c},\qquad
V_{t,c}=P_{t,2C+c}
\quad(0\le c<C).
$$

## 3. Compute causal multi-head attention

Head $h\in\{0,\ldots,H-1\}$ uses channels $hd,\ldots,(h+1)d-1$ from each of $Q,K,V$. Write these length-$d$ slices as $q_{t,h},k_{t,h},v_{t,h}$.

At query position $t$, only key positions $j\le t$ are constructed. Their scaled dot-product scores are

$$
s_{h,t,j}
=\frac{q_{t,h}\cdot k_{j,h}}{\sqrt d}
=\frac{1}{\sqrt d}\sum_{r=0}^{d-1}
Q_{t,hd+r}K_{j,hd+r},
\qquad 0\le j\le t.
$$

Softmax turns those $t+1$ scores into weights:

$$
m_{h,t}=\max_{0\le j\le t}s_{h,t,j},\qquad
p_{h,t,j}
=\frac{\exp(s_{h,t,j}-m_{h,t})}
{\sum_{u=0}^{t}\exp(s_{h,t,u}-m_{h,t})}.
$$

Subtracting the maximum improves numerical stability without changing the mathematical probabilities. Equivalently, future positions $j>t$ have probability zero. For a three-token sequence, the allowed keys are \(\{0\}\), \(\{0,1\}\), and \(\{0,1,2\}\) for queries 0, 1, and 2 respectively.

The head output is a weighted sum of value vectors:

$$
a_{t,h}=\sum_{j=0}^{t}p_{h,t,j}v_{j,h},
\qquad
A_{t,hd+r}=(a_{t,h})_r
\quad(0\le r<d).
$$

Writing each $a_{t,h}$ into its own channel range concatenates the heads into $A\in\mathbb{R}^{T\times C}$. Heads use separate channel slices here; the next projection can mix them.

## 4. Project attention and add the first residual

$$
O=\operatorname{Linear}(A;W_{att},b_{att}),
\qquad W_{att}\in\mathbb{R}^{C\times C},
$$

$$
R=X+O.
$$

The addition is elementwise over positions and channels. Notice that the bypass uses the original $X$, not the normalized $N^{(1)}$.

## 5. Normalize for the MLP

The same LayerNorm formula is applied to each row of $R$, now with the second set of learned parameters:

$$
N^{(2)}=\operatorname{LayerNorm}(R;\gamma^{(2)},\beta^{(2)},10^{-5}).
$$

`ln2w` and `ln2b` hold $\gamma^{(2)}$ and $\beta^{(2)}$. The residual stream $R$ stays available for the final addition.

## 6. Expand, apply GELU, and project back

The MLP is applied independently to each token row; it does not mix token positions. First it expands $C$ channels to $4C$:

$$
F=\operatorname{Linear}(N^{(2)};W_{fc},b_{fc}),
\qquad W_{fc}\in\mathbb{R}^{4C\times C},
\qquad F\in\mathbb{R}^{T\times 4C}.
$$

Then `gelu()` applies GPT-2's tanh approximation elementwise:

$$
G_{t,i}=\frac{F_{t,i}}{2}
\left[1+\tanh\!\left(
\sqrt{\frac{2}{\pi}}
\left(F_{t,i}+0.044715F_{t,i}^{3}\right)
\right)\right].
$$

Finally, another linear projection returns to $C$ channels:

$$
M=\operatorname{Linear}(G;W_{proj},b_{proj}),
\qquad W_{proj}\in\mathbb{R}^{C\times 4C},
\qquad M\in\mathbb{R}^{T\times C}.
$$

For GPT-2 124M, the MLP width is $4C=3072$. The checkpoint names for these two projections are `fcw`/`fcb` and `fcprojw`/`fcprojb`.

## 7. Add the second residual

$$
\boxed{Y=R+M}
$$

`Y` becomes the input to the next Transformer layer. The whole block can be summarized as

$$
\begin{aligned}
R &= X+\operatorname{AttentionProjection}\!\left(
\operatorname{CausalMultiHeadAttention}\!\left(
\operatorname{QKV}\!\left(\operatorname{LayerNorm}_1(X)\right)
\right)\right),\\
Y &= R+\operatorname{MLP}\!\left(\operatorname{LayerNorm}_2(R)\right).
\end{aligned}
$$

The returned `inspected_scores` and `inspected_probabilities` are diagnostics from **head 0 at the last query position**. They have length $T$ and do not alter $Y$. The current implementation recomputes attention over the full context and does not apply dropout during inference.
