# Mathematics of `transformer_block`

This document follows [`transformer_block`](../src/main.cpp) in execution order. It describes **one GPT-2 Transformer layer** during inference. The input already contains token and position embeddings; the final LayerNorm and vocabulary projection happen after all Transformer blocks.

## The key relationship: embedding dimension = block width

GPT-2 chooses one model width, called $C$ here. In this implementation, **the token embedding dimension, position embedding dimension, and channel count of the residual stream are all that same $C$ by design**. For GPT-2 124M, $C=768$. It is not a numerical coincidence, and the channel count is not a multiple calculated from a separate embedding dimension.

The checkpoint header stores `channel_count`. The code uses it to define the token-embedding table `wte` with shape $V_{\mathrm{pad}}\times C$ and the position-embedding table `wpe` with shape $T_{\max}\times C$. Here $V_{\mathrm{pad}}$ is the padded table row count, and $T_{\max}$ is the maximum number of positions. Looking up one row of each table gives two vectors of length $C$, which the code adds to create the first hidden state at position $t$:

$$
x_t=\operatorname{wte}[\text{token\_id}_t,:]
+\operatorname{wpe}[t,:]\in\mathbb{R}^{C}.
$$

A block receives $T$ such rows and returns $T$ rows of the same width:

$$
X^{(\ell)}\in\mathbb{R}^{T\times C}
\xrightarrow{\text{Transformer block }\ell}
X^{(\ell+1)}\in\mathbb{R}^{T\times C}.
$$

Keeping that width lets each block add its attention and MLP results to the residual stream. The vocabulary size $V$ counts **how many valid token IDs** have embeddings; $C$ counts **how many numbers are in each embedding**. The number of heads $H$ is another separate choice, with the constraint that $H$ divides $C$ so each head has $d=C/H$ channels. A model could be designed with a different embedding width and an extra conversion projection, but this GPT-2 implementation does not have one.

## First, what is a channel?

For each token position, the model keeps a list of numbers representing that token at the current stage of computation. One entry in that list is a **channel**, also called a feature or hidden dimension. Here the list has the model width $C=768$ entries, so the residual stream has 768 channels. `ActivationMatrix(position, channel)` reads one such number.

For example, if the context has three tokens and four channels in a toy model, its activations might look like this:

$$
X=\begin{bmatrix}
 0.2 & -0.1 & 0.8 & 0.4 \\
 0.5 &  0.3 & 0.1 & -0.2 \\
-0.4 &  0.7 & 0.6 & 0.9
\end{bmatrix}
\quad\text{with shape }3\times4.
$$

The second row is the four-number representation of the token at position 1. The value $X_{1,2}=0.1$ is channel 2 at that position. A channel is **a coordinate of a learned representation**, not a character, a word, an attention head, or a human-assigned concept. Its meaning depends on the learned weights and on the other channels. The same channel index exists at every token position, but its value can differ at each position. The initial row comes from token and position embeddings; after attention, the row can also reflect earlier tokens.

The 768-channel width is maintained at the input and output of every GPT-2 block so residual additions can add two rows element by element. Inside a block, some intermediate vectors have different widths: Q, K, and V each have 768 channels; their combined projection has $3\cdot768=2304$ channels; the MLP temporarily expands to $4\cdot768=3072$ channels. These are widths of intermediate representations, not extra tokens.

## Terms used in the equations

### Reading the notation

- **Scalar, vector, matrix, tensor:** A scalar is one number. A vector is a one-dimensional list of numbers, such as one token's hidden state. A matrix is a two-dimensional grid of numbers, such as all token rows together. *Tensor* is the general name for an array with any number of dimensions; in this document, the activations are mostly matrices.
- **Shape / dimension / width:** The shape lists an array's sizes along each axis. A $T\times C$ activation matrix has $T$ rows (token positions) and $C$ columns (channels). Its width is $C$. For the toy matrix above, $T=3$ and $C=4$. This implementation has no separate batch axis in `ActivationMatrix`.
- **Index / subscript:** $X_{t,c}$ means the number in row $t$, column $c$ of $X$. The first index selects a position; the second selects a channel. All indices in this document start at zero.
- **Elementwise operation:** Apply an operation to matching entries. For example, $(X+O)_{t,c}=X_{t,c}+O_{t,c}$. This requires $X$ and $O$ to have the same shape.
- **Dot product:** Multiply corresponding entries of two equal-length vectors and add the products. For example, $(a_0,a_1)\cdot(b_0,b_1)=a_0b_0+a_1b_1$. Attention uses this to compare a query with a key.
- **Weighted sum:** Multiply each value vector by its weight, then add the results. In attention, the weights are softmax probabilities and the vectors are values from allowed token positions.

### Input and representation

- **Token:** A unit produced by GPT-2's tokenizer. It can represent a word, part of a word, punctuation, or other text bytes. One token ID selects one row from the token-embedding table.
- **Position:** A token's zero-based place in the current input sequence. If there are $T$ tokens, the positions are $0,\ldots,T-1$. This is the row index of `ActivationMatrix`.
- **Context / sequence:** The tokens supplied together to the model for a forward pass. Here `transformer_block` processes all $T$ positions. A causal rule controls which earlier positions each position can use.
- **Embedding:** A learned vector looked up for a token ID or a position. Before the first block, the code adds token and position embeddings to form each input row. Both embedding vectors have width $C$, the same as the block input and output; their individual coordinates are channels.
- **Activation:** A number computed during the forward pass, as opposed to a stored model parameter. `input`, normalized rows, Q/K/V, attention outputs, and MLP outputs are all activations. `ActivationMatrix` stores a matrix of them.
- **Hidden state:** The current vector of activations for a token position. The full $T\times C$ matrix contains one hidden-state row per position. These rows change as they pass through successive blocks.
- **Residual stream:** The $T\times C$ matrix carried from block to block. Each block adds attention and MLP results to this stream. The variable `input` is the stream entering this block; `post_attention` is the stream after the first addition.

### Model structure and operations

- **Layer / Transformer block:** One repeated unit of computation. This implementation's block contains attention and an MLP, each preceded by LayerNorm and followed by a residual addition. GPT-2 124M repeats it 12 times; `layer_index` chooses one unit's parameters.
- **Parameter / weight / bias:** A number learned during training and saved in the checkpoint. A weight matrix determines how input channels contribute to output channels. A bias vector adds one learned offset per output channel. During this inference pass, they are read but not updated.
- **Linear projection:** For each token row, multiply by a weight matrix and add a bias. Despite the name, the bias makes it an affine operation. A projection can change the width, such as $C\to3C$ or $C\to4C$, or keep it at $C\to C$. It mixes channels within a row; by itself, it does not mix token positions.
- **LayerNorm:** For each token row separately, compute the mean and variance across its channels, normalize the row, and apply learned scale $\gamma$ and offset $\beta$. It does not average across tokens. The small $\varepsilon$ inside the square root prevents division by zero or a very small number.
- **Pre-normalization:** Normalize the input *before* sending it to attention or the MLP. The unnormalized residual stream goes around each operation and is used in the addition. This is why the first residual is $X+O$, rather than $N^{(1)}+O$.
- **Residual connection / skip connection:** Add an operation's output to the stream that entered that part of the block. This block has two: `input + projected_attention` and `post_attention + projected_mlp`. Both summands must have shape $T\times C$.
- **MLP / feed-forward network:** The two linear projections with a GELU activation between them. It transforms each token row independently. Here the first projection expands $C\to4C$ and the second contracts $4C\to C$.
- **GELU:** The non-linear activation applied to each MLP value. Unlike a linear projection, it cannot be collapsed into one matrix multiplication with the neighboring projections. The code uses a tanh approximation shown in step 6.

### Attention

- **Self-attention:** A way for a token position to combine information from other positions in the *same* sequence. The position's query is compared with keys, and the resulting weights select a weighted mixture of values. This is where token positions interact inside the block.
- **Query (Q):** What a position uses to score possible source positions. Each position has a query vector for every head.
- **Key (K):** What a possible source position presents for comparison with a query. The dot product of a query and key becomes an attention score.
- **Value (V):** The vector of information a source position contributes after the scores become attention probabilities. Keys determine *how much* to read; values determine *what vector* gets mixed into the result. Q, K, and V are different learned projections of the same normalized input.
- **Attention head:** One separate attention calculation over a slice of the Q, K, and V channels. With $H=12$ heads and $C=768$ channels, each head works with $d=C/H=64$ Q channels, 64 K channels, and 64 V channels. Its output has 64 channels. The outputs of all heads are concatenated into 768 channels.
- **Attention score:** The query-key dot product divided by $\sqrt d$. It is a raw compatibility number, not yet a probability. The scaling keeps dot products from growing too large as $d$ grows.
- **Softmax / attention probability:** Softmax exponentiates and normalizes the allowed scores so each probability is nonnegative and the probabilities for one query and head sum to 1. These probabilities weight the value vectors.
- **Causal mask:** The rule that a query at position $t$ can use keys and values at positions $0,\ldots,t$, but not later positions. The code enforces this by never constructing scores for future positions. Thus the last position can see the whole context, while the first can see only itself.
- **Multi-head attention:** Run the attention calculation separately for each head, concatenate their outputs, then use the attention output projection to mix channels across heads.

### Scope of this block

- **Inference:** Use trained parameters to compute outputs. This block performs a forward pass; it has no gradients or parameter updates.
- **Logit:** An unnormalized score for a vocabulary token. Logits are computed *after* all Transformer blocks and the final LayerNorm, so they are different from the attention scores inside this document.

## A small attention example

Suppose a toy model has $C=4$ channels and $H=2$ heads, so each head has $d=2$ channels. Head 0 uses channels 0 and 1; head 1 uses channels 2 and 3. At query position $t=1$, neither head may read position 2.

For head 0, imagine $q_{1,0}=(1,0)$, $k_{0,0}=(1,0)$, and $k_{1,0}=(0,1)$. The allowed scores are $s_{0,1,0}=1/\sqrt2$ and $s_{0,1,1}=0$. Softmax makes their probabilities approximately $0.67$ and $0.33$. If $v_{0,0}=(2,0)$ and $v_{1,0}=(0,4)$, the output is approximately

$$
a_{1,0}=0.67(2,0)+0.33(0,4)=(1.34,1.32).
$$

This two-number output fills head 0's part of position 1's attention row. Head 1 fills the remaining two channels. The full four-channel row then goes through the attention output projection and is added to the original input row.

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
