/* TODOS-
 * - Style links
 * - Eq numbering
 * - Sec numbering
 * - Fix algos
 * - TOC styling
 * - Multi-citation fmt
 * */


#import "@preview/lovelace:0.3.0": *

/* Custom objects */
#let DR = `Dropout`
#let SM = `Softmax`
#let CAT = `Concat`
#let LIN = `Linear`
#let LN = `LayerNorm`
#let MLP = `MLP`
#let CA = `CausalAttention`
#let NORM = `Norm`
#let CUMSUM = `cumsum`
#let SEGSUM = `segsum`
#let SUM = `sum`
#let MHA = `MHA`
#let TOPK = `top_k`
#let TR = `Trace`

#set table(
  inset: 6pt,
  stroke: none,
)

#show figure.where(kind: table): set figure.caption(position: top)

#show figure.where(kind: image): set figure.caption(position: bottom)

#let content-to-string(content) = {
  if content.has("text") {
    content.text
  } else if content.has("children") {
    content.children.map(content-to-string).join("")
  } else if content.has("body") {
    content-to-string(content.body)
  } else if content == [ ] {
    " "
  }
}
#let conf(
  title: "Decoder-Only Transformers",
  subtitle: none,
  authors: "Garrett Goon",
  keywords: (),
  date: none,
  abstract: none,
  cols: 1,
  margin: (x: 1.0in, y: 1.0in),
  paper: "us-letter",
  lang: "en",
  region: "US",
  font: (),
  fontsize: 12pt,
  sectionnumbering: "1.",
  pagenumbering: "1",
  doc,
) = {
  set document(
    title: title,
    author: authors.map(author => content-to-string(author.name)),
    keywords: keywords,
  )
  set page(
    paper: paper,
    margin: margin,
    numbering: pagenumbering,
    columns: cols,
  )
  set par(justify: true)
  set text(
    lang: lang,
    region: region,
    font: font,
    size: fontsize,
  )
  set heading(numbering: sectionnumbering)

  place(top, float: true, scope: "parent", clearance: 4mm)[
    #if title != none {
      align(center)[#block(inset: 2em)[
          #text(weight: "bold", size: 1.5em)[#title]
          #(
            if subtitle != none {
              parbreak()
              text(weight: "bold", size: 1.25em)[#subtitle]
            }
          )
        ]]
    }

    #if authors != none and authors != [] {
      let count = authors.len()
      let ncols = calc.min(count, 3)
      grid(
        columns: (1fr,) * ncols,
        row-gutter: 1.5em,
        ..authors.map(author => align(center)[
          #author.name \
          #author.affiliation \
          #author.email
        ])
      )
    }

    #if date != none {
      align(center)[#block(inset: 1em)[
          #date
        ]]
    }

    #if abstract != none {
      block(inset: 2em)[
        #text(weight: "semibold")[Abstract] #h(1em) #abstract
      ]
    }
  ]

  doc
}
#show: doc => conf(
  title: [Decoder-Only Transformers],
  authors: (
    (name: [Garrett Goon], affiliation: "", email: ""),
  ),
  abstract: [Notes on various aspects of Decoder-Only Transformers. Conventions are in the appendix.
  ],
  pagenumbering: "1",
  cols: 1,
  doc,
)

/* Outline styling */
#show outline.entry.where(level: 1): it => {
  v(12pt, weak: true)
  strong(it)
}
#outline(indent: auto)

/* Eq ref styling https://typst.app/docs/reference/model/ref/ */
#set math.equation(numbering: "(1)", number-align: end + bottom)
#show ref: it => {
  let foot = footnote
  let eq = math.equation
  let el = it.element
  if el != none and el.func() == eq {
    // Override equation references.
    link(
      el.location(),
      numbering(
        el.numbering,
        ..counter(eq).at(el.location()),
      ),
    )
  } else if el != none and el.func() == foot {
    // Other references as usual.
    link(
      el.location(),
      "Footnote "
        + numbering(
          el.numbering,
          ..counter(eq).at(el.location()),
        ),
    )
  } else {
    // Other references as usual.
    it
  }
}


= Architecture
<architecture>

== Decoder-Only Fundamentals
<sec_decoder_only>

The Transformers architecture @vaswani2017attention, which dominates
Natural Language Processing (NLP) as of July 2023, is a relatively
simple architecture. There are various flavors and variants of
Tranformers, but focus here on the decoder-only versions which underlie
the GPT models
@gpt2radford2019language@gpt3brown2020language@gpt4openai2023.

The full decoder-only architecture can be seen in
Fig.~@fig_transformers_architecture. The parameters which define the
network can be found in #link(<app_conventions>)[Conventions].

#figure(
  image("figures/transformer-general.jpg"),
  caption: [
    The full transformers architecture. Diagram taken from
    @korthikanti2022reducing
  ],
)
<fig_transformers_architecture>

At a high level, decoder-only transformers take in an ordered series of
word-like objects, called tokens, and are trained to predict the next
token in the sequence. Given some initial text, transformers can be used
to give a prediction for the likelihood of any possible continuation of
that text. An outline of the mechanics#footnote[This describes the
vanilla architecture; almost every component is modified in the
available variants.]:

+ Raw text is #strong[tokenized] and turned into a series of
  integers#footnote[There are about
  #link("https://github.com/ray-project/llm-numbers")[1.3 tokens per word],
  on average.] whose values lie in `range(V)`, with $V$ the vocabulary size.

+ The tokenized text is chunked and turned into `(B, S)`-shaped (batch size and
  sequence length, respectively) integer tensors, $x_(b s)$.

+ The #strong[embedding layer] converts the integer tensors into
  continuous representations of shape `(B, S, D)`, $z_(b s d)$, with $D$ the size
  of the hidden dimension. #strong[Positional encodings] have also been
  added to the tensor at this stage to help the architecture understand
  the relative ordering of the text.

+ The $z_(b s d)$ tensors pass through a series of transformer blocks,
  each of which has two primary components:

  + In the #strong[attention] sub-block, components of $z_(b s d)$ at
    different positions ($s$-values) interact with each other, resulting
    in another `(B, S, D)`-shaped tensor, $z'_(b s d)$.

  + In the #strong[MLP] block, each position in $z'_(b s d)$ is
    processed independently and in parallel by a two-layer feed-forward
    network, resulting once more in a -shaped tensor.

  Importantly, there are #strong[residual connections] around each of
  these#footnote[This gives rise to the concept of the #strong[residual
  stream] which each transformer block reads from and writes back to
  repeatedly.] (the arrows in Fig.~@fig_transformers_architecture),
  meaning that the output of each block is added back to its original
  input.

+ Finally, we convert the `(B, S, D)`-shaped tensors to `(B, S, V)`-shaped ones, $y_(b s v)$.
  This is the role of the #strong[language model head] (which is often
  just the embedding layer used in an inverse manner.)

+ The $y_(b s v)$ predict what the next token will be, i.e.
  $x_(b s + 1)$, having seen the #strong[context] of the first $s$
  tokens in the sequence. Specifically, removing the batch index for
  simplicity, a $SM$ of $y_(s v)$ gives the conditional probability
  $p_(s v) = P (t_(s + 1) \| t_s dots.h t_0)$ for the indicated series
  of tokens. Because of the chain rule of probability, these individual
  probabilities can be combined to form the probability that any
  sequence of tokens follows a given initial seed#footnote[In more
  detail, these probabilities are created by products:
  $P (t_(s + n) dots.h t_(s + 1) \| t_s dots.h t_0) = P (t_(s + n) \| t_(s + n - 1) dots.h t_s dots.h t_0) times dots.h times P (t_(s + 1) \| t_s dots.h t_0)$.].

Each batch (the $b$-index) is processed independently. We omitted $LN$ and $DR$
layers above, as well as the causal mask; these will be covered below as
we step through the architecture in more detail.

=== Embedding Layer and Positional Encodings <subsubsec_embedding_and_pe>
The #strong[embedding] layer is just a simple lookup table: each of the `range(V)` indices in the
vocabulary is mapped to a $D$-dimensional vector via a large `(V, D)`-shaped table/matrix. This layer maps
$x_(b s) arrow.r z_(b s d)$. In , this is an `nn.Embedding(V, D)` instance.

To each item in a batch, we add identical #strong[positional encodings] to the vectors above with
the goal of adding fixed, position-dependent correlations in the sequence dimension which will
hopefully make it easier for the architecture to pick up on the relative positions of the inputs
#footnote[Positional encodings and the causal mask are the only components in the vanilla
  transformers architecture which carry weights with a dimension of size $S$; i.e. they are the only
  parts that have explicit sequence-length dependence. A related though experiment: you can convince
  yourself that if the inputs $z_(b s d)$ were just random noise, the transformers architecture
  would not be able to predict the $s$-index of each such input in the absence of positional
  encodings.] This layer maps $z_(b s d) arrow.l z_(b s d) + p_(s d)$, with $p_(s d)$ the positional
encoding tensor.

The above components require $(V + S) D approx V D$ parameters per
model.

=== Layer Norm <layer_norm>
The original transformers paper @vaswani2017attention put $LN$ instances
after the #strong[attention] and #strong[MLP] blocks, but now it is
common @xiong2020layer to put them before these blocks#footnote[Which
makes intuitive sense for the purposes of stabilizing the matrix
multiplications in the blocks].

The $LN$ operations acts over the hidden dimension (since this is the
dimension the subsequent $LIN$ instances act on). Spelling it out, given the
input tensor $z_(b s d)$ whose mean and variance over the $d$-index are
$mu_(b s)$ and $sigma_(b s)$, respectively, the $LN$ output is
$
  z_( b s d ) & <- ( (z_( b s d ) - mu_( b s ) ) / sigma_( b s ) ) gamma_d
  + beta_( d ) equiv LN_d z_( b s d )
$
where $gamma_d \, beta_d$ are the trainable scale and
bias parameters. In `torch`, this is a `nn.LayerNorm(D)` instance. Since there are two $LN$ instances
in each transformer block, these components require $2 D$ parameters per
layer.


We will continue discussing $LN$ instances in what follows in order to adhere to the usual
construction and to discuss methods like sequence-parallelism in their original form (see
@subsec_seq_parallelism), but note: the data-independent $LN$ transformations due to $gamma_d \, beta_d$
are completely redundant when immediately followed by a $LIN$ layer, since both act linearly on their
inputs and $LIN$ is already the most general data-independent linear transformation. Explicitly, the
$gamma_d \, beta_d$ parameters can be absorbed into the $LIN$ parameters:
$
  (x_(b s d) gamma_d + beta_d) W_(d d') + b_(d') & = x_(b s d) W'_(d d') + b'_(d') med \, quad W'_(d d') equiv gamma_d W_(d d') med \, quad b'_(d') equiv b_(d') + beta_d W_(d d') med \,
$
for arbitrary $x_(b s d)$. That is, these transformations can be
equivalently performed by the weight matrix and bias (if included) in the $LIN$
layer#footnote[Note the importance of data-independence here: the
data-dependent mean and standard deviation terms cannot be similarly
absorbed. Also, because the usual training algorithms are not invariant
under parameter redefinitions, the above unfortunately does not imply
that removing the $LIN$ learnable parameters (`elementwise_affine=False` in `torch`) will have no effect on
training dynamics. $gamma_d \, beta_d$ can shoved into the $LIN$ layer's
parameters as a small inference-time optimization, though.].

=== Causal Attention <attn_layer>
#strong[Causal attention] is the most complex layer. It features $A$
sets of weight matrices#footnote[There are also bias terms, but we will
often neglect to write them explicitly or account for their (negligible)
parameter count.] $Q_(d e a) \, K_(d e a) \, V_(d e a)$ where
$a in {0 \, dots.h \, A - 1}$ and $e in {0 \, dots.h \, D \/ A}$, where
$D$ is assumed perfectly divisible by $A$. From these, we form three
different vectors:
$
  q_(b s e a) & = z_(b s d) Q_(d e a) med \, quad k_(b s e a) = z_(b s d) K_(d e a) med \, quad v_(b s e a) = z_(b s d) V_(d e a)
$
These are the #strong[query, key, and value] tensors, respectively
#footnote[There are of course many variants of the architecture and one
variant which is popular in Summer 2023 is multi-query attention
/* @shazeer2019fast in which all heads share #emph[the same] key and value */
vectors and only the query changes across heads, as this greatly reduces
/* inference costs. See @subsec_multi_query_attn.  */

 ].

Using the above tensors, we will then build up an #strong[attention map]
$w_(b s s' a)$ which corresponds to how much attention the token at
position $s$ pays to the token at position $s'$. Because we have the
goal of predicting the next token in the sequence, we need these weights
to be causal: the final prediction $y_(b s v)$ should only have access
to information propagated from positions $x_(b s' v)$ with $s' lt.eq s$.
This corresponds to the condition that $w_(b s s' a) = 0$ if $s' > s$.
The entire causal Transformers architecture as a whole obeys this
condition: the outputs
$z_(b s d) = mono("CausalTransformer") (x_(b s' d'))$ only depend on
those inputs $x_(b s' d')$ with $s' lt.eq s$.

These weights come from $SM$-ed attention scores, which are just a
normalized dot-product over the hidden dimension:
$
  w_( b s s' d a ) & =SM_( s' ) (m_( s s' )+(q_( b s e )k_( b s' e a ) )( sqrt(D / A)) ), "s.t." sum_(s')w_( b d s s' a ) =1
$

The tensor $m_(s s')$ is the causal mask which zeroes
out the relevant attention map components above
$
  m_{ s s\' } & = cases(
  0 & s <= s' ,
  - infinity & = s > s'
)
$
forcing $w_(b s s' d a) = 0$ for $s > s'$. In other words, the causal mask ensures that a given
tensor, say $z_(b s d)$, only has dependence on other tensors whose sequence index, say $s'$, obeys
$s' lt.eq s$. This is crucial for inference-time optimizations, in particular the use of the
#strong[kv-cache] in which key-value pairs do not need to be re-computed.

The $sqrt(D \/ A)$ normalization is motivated by demanding that the variance of the $SM$ argument be 1 at
initialization, assuming that other components have been configured so that that the query and key
components are i.i.d. from a Gaussian normal distribution .

The weights above are then passed through a dropout layer and used to
re-weigh the #strong[value] vectors and form the tensors
$
  y_( b s e a) & = DR (w_( b d s s' a) ) v_( b s'e a )
$<eq_reweighted_values>
and these `(B, S, D/A, A)`-shaped tensors are then concatenated along
the $e$-direction to re-form a `(B, S, D)`-shaped tensor $u_(b s d)$
$ u_(b s d) & = y_(b s (e a)) $ in
#link("https://einops.rocks/1-einops-basics/")[`einops`]-like notation for
concatenation. Finally, another weight matrix $O_(d' d)$ and dropout
layer transform the output once again to get the final output
$
  z_( b s d ) & = DR (u_( b s d' ) O_( d'd ) ) .
$


For completeness, the entire operation in condensed notation with
indices left implicit is:
$
  z & arrow DR ( CAT ( DR (SM ( ( ( z dot Q_( a ) )dot ( z dot K_( a ) )) / sqrt(D / A) ) )dot z dot V_( a ) ) dot O )
$<eq_causal_attn>
where all of the dot-products are over feature
dimensions (those of size $D$ or $D \/ A$).

Below is pedagogical#footnote[The code is written for clarity, not
speed. An example optimization missing here: there is no need to form
separate $Q_a \, K_a \, V_a$ $LIN$ layers, one large layer which is later
chunked is more efficient] sample code for such a $CA$ layer#footnote[When
using sequence-parallelism, it will be more natural to separate out the
final $DR$ layer and combine it with the subsequent $LN$, as they are sharded
together; see @subsec_seq_parallelism. The same is true for the $MLP$
layer below.]:

```python
class CausalAttention(nn.Module):
    def __init__(
        self,
        block_size=K,
        dropout=0.1,
        hidden_dim=D,
        num_attn_heads=A,
    ):
        super().__init__()
        self.block_size = block_size
        self.dropout = dropout
        self.hidden_dim = hidden_dim
        self.num_attn_heads = num_attn_heads

        self.head_dim, remainder = divmod(hidden_dim, num_attn_heads)
        assert not remainder, "num_attn_heads must divide hidden_dim evenly"

        self.Q = nn.ModuleList(
            [nn.Linear(hidden_dim, self.head_dim) for _ in range(num_attn_heads)]
        )
        self.K = nn.ModuleList(
            [nn.Linear(hidden_dim, self.head_dim) for _ in range(num_attn_heads)]
        )
        self.V = nn.ModuleList(
            [nn.Linear(hidden_dim, self.head_dim) for _ in range(num_attn_heads)]
        )
        self.O = nn.Linear(hidden_dim, hidden_dim)

        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(block_size, block_size)[None]),
        )

    def get_qkv(self, inputs):
        queries = [q(inputs) for q in self.Q]
        keys = [k(inputs) for k in self.K]
        values = [v(inputs) for v in self.V]
        return queries, keys, values

    def get_attn_maps(self, queries, keys):
        S = queries[0].shape[1]
        norm = math.sqrt(self.head_dim)
        non_causal_attn_scores = [(q @ k.transpose(-2, -1)) / norm for q, k in zip(queries, keys)]
        # Note: this mask shape is a bit of a hack to make generation from the KV cache work without
        # specifying an extra boolean. When queries and keys have different sequence lengths and the
        # queries are of seq_len == 1,p the query attends to all of the keys; effectively there is
        # no mask at all.
        causal_attn_scores = [
            a.masked_fill(self.causal_mask[:, :S, :S] == 0, float("-inf"))
            for a in non_causal_attn_scores
        ]
        attn_maps = [a.softmax(dim=-1) for a in causal_attn_scores]
        return attn_maps

    def forward(self, inputs):
        queries, keys, values = self.get_qkv(inputs)
        attn_maps = self.get_attn_maps(queries, keys)
        weighted_values = torch.cat(
            [self.attn_dropout(a) @ v for a, v in zip(attn_maps, values)], dim=-1
        )
        z = self.O(weighted_values)
        z = self.out_dropout(z)
        return z
```

The parameter count is dominated by the weight matrices which carry
$4 D^2$ total parameters per layer.

=== MLP <subsubsec_mlp>
The feed-forward network is straightforward and corresponds to
$
  z_( b s d ) & -> DR (phi ( z_( b s d' )W^0_( d'e ) ) W^1_( e d ) )
$<eq_mlp>
where $W^0$ and $W^1$ are `(B, S, D)`- and `(E*D, D)`-shaped matrices,
respectively (see App.~@app_conventions for notation) and $phi$ is a
non-linearity#footnote[The `GeLU`
#link("https://pytorch.org/docs/stable/generated/torch.nn.GELU.html")[non-linearity]
is common.]. In code, where we again separate out the last $DR$ layer as we
did in in @attn_layer:
```python
class MLP(nn.Module):
    def __init__(
        self,
        hidden_dim=D,
        expansion_factor=E,
        dropout=0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.expansion_factor = expansion_factor
        self.dropout = dropout

        linear_1 = nn.Linear(hidden_dim, expansion_factor * hidden_dim)
        linear_2 = nn.Linear(expansion_factor * hidden_dim, hidden_dim)
        gelu = nn.GELU()
        self.layers = nn.Sequential(linear_1, gelu, linear_2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        z = self.layers(inputs)
        z = self.dropout(z)
        return z
```

This bock requires $2 E D^2$ parameters per layer, only counting the
contribution from weights.

=== Language Model Head <subsubsec_language_model_head>
The layer which converts the `(B, S, D)`-shaped outputs, $z_(b s d)$, to `(B, S, V)`-shaped
predictions over the vocabulary, $y_(b s v)$, is the #strong[Language
Model Head]. It is a linear layer, whose weights are often tied to be
exactly those of the initial embedding layer of
@subsubsec_embedding_and_pe.

=== All Together
<all-together>
It is then relatively straightforward to tie every thing together. In
code, we can first create a transformer block like which corresponds to
the schematic function
$
  z & arrow z + MLP ( LN ( z + CA (LN ( z ) ) ))
$
indices suppressed.


=== The Loss Function
<the-loss-function>
The last necessary component is the loss function. The training loop data is the `(B, K)`-shaped#footnote[`K` is
the block size, the maximum sequence-length for the model. See App.~@app_conventions.] token
inputs ($x_(b s)$) along with their shifted-by-one relatives $y_(b s)$ where `x[:, s + 1] == y[:, x]` . The `(B, K, V)`-shaped outputs
($z_(b s v)$) of the `DecoderOnly` network are treated as the logits which predict the value of the next token,
given the present context:
$
  p(x_( b (s+1) )=v| x_( b s ), x_( b (s-1) ), ..., x_( b 0 )) & = SM_( v ) z_( b s v )
$
<eq_transformer_conditional_prob>


and so the model is trained using the usual
cross-entropy/maximum-likelihood loss#footnote[Here's an alternative
derivation for why this loss is minimized when the learned distribution
perfectly matches the actual one. Let $p (x)$ be the actual distribution
and $q_theta (x)$ be the model. Taking the continuous case, the expected
loss is $cal(L) = - integral dif x thin p(x)ln q _(
    theta)(x)$. We want to minimize this, subject to the condition
that $integral dif x q _( theta
    )(x) =1$. So, we use the
#link("https://e n.wikipedia.org/wiki/Calculus_of_variations")[calculus of variations]
on the loss with a Lagrange multiplier:
$cal(L)' = cal(L) + lambda integral dif x thin q_( theta )(x)$. Solving
$( delta cal(L)' )/( delta q _( theta )(x) )=0$ yields
$q_theta (x) = p (x)$. This seems more straightforward and general than
the usual argument via the KL-divergence and Jensen's inequality.]
$
  cal(L) & = -1 / (B K) sum_( b,s )ln p(x_( b (s+1) )=y_( b(s+1) )| x_( b s ), x_( b (s-1) ),
    ..., x_( b 0 )) \
  & = - 1 / ( B K )sum_( b,s )SM_( v ) z_( b s v)|_( v=y_( b(s+1) ) ) .
$
Note that the losses for all possible context lengths
are included in the sum, equally weighted#footnote[In Natural Language
Processing (NLP), the perplexity is often reported instead of the loss, which is
just the exponential of the loss, a geometric-mean over the gold-answer
probabilities:
$"perplexity" = e^( cal(L) ) = (product_( b, s )p(x _( b
        (s+1) )=| x _( b s ), x _( b (s-1) ), ..., x _( b 0 )) ) ^(  -1 /( B K ) )$.].

In `torch` code, the loss computation might look like the following (using fake data):
```python
model = DecoderOnly(
    num_attn_heads=A,
    block_size=K,
    dropout=0.1,
    expansion_factor=E,
    hidden_dim=D,
    num_layers=L,
    vocab_size=V,
)
tokens = torch.randint(model.vocab_size, size=(B, model.block_size + 1))
inputs, targets = tokens[:, :-1], tokens[:, 1:]
outputs = model(inputs)
outputs_flat, targets_flat = outputs.reshape(-1, outputs.shape[-1]), targets.reshape(-1)
loss = F.cross_entropy(outputs_flat, targets_flat)
```

== Architecture and Algorithm Variants
<architecture-and-algorithm-variants>
There are, of course, many variants on the basic architecture. Some
particularly important ones are summarized here.

=== GLU Variants<subsec_glu_variants>
In @shazeer2020gluvariantsimprovetransformer, Shazeer advocated for
replacing the usual linear-then-activation function pattern,
$ z_(d') & = phi (W_(d' d) x_d) $ to
$ z_(d') & = V_(d' e) x_e phi (W_(d' d) x_d) med . $ So, just
perform another linear operation on the original input and broadcast it
against the usual activation function output. Biases for can also be
included. This construction is typically called “$phi$GLU\" where
$phi$ is the name of the activation function: ReGLU, SwiGLU/SiGLU
($phi = x sigma (x)$ used in the LLaMA models), etc.

=== Multi-Query Attention <subsec_multi_query_attn>
In @shazeer2019fast, the $A$ different key and value matrices are
replaced by a single matrix each, while $A$ different query-heads
remain. The mechanisms are otherwise unchanged: where there were
previously distinct key and value tensors used across different heads,
we just use the same tensors everywhere. This is #strong[Multi-Query
Attention] (MQA).

The primary reason for multi-query attention is that it vastly reduces
the size of the kv-cache (see @sec_kv_cache) during inference time,
decreasing the memory-burden of the cache by a factor of $A$. This
strategy also reduces activation memory during training, but that is
more of a side-effect.

=== Grouped Attention <subsec_grouped_attn>
#strong[Grouped Query Attention] (GQA) @ainslie2023gqa is the natural
extension of multi-query-attention to using $1 < G < A$ matrices for key
and value generation. Each of the $G$ different keys gets matched up
with $A \/ G$ heads (nice divisibility assumed)#footnote[Llama-2
@touvron2023llama2 uses GQA with $G = 8$, seemingly chosen so that each
group can be sharded and put on its own GPU within a standard 8-GPU
node.].

=== Parallel $MLP$ and $CA$ Layers
<parallel-and-layers>
Rather than first pass inputs into the $CA$ layer of each block, and then
pass those outputs on to $MLP$ in series,
#link("https://github.com/kingoflolz/mesh-transformer-jax/blob/f8315e3003033b23f21d78361b288953064e0e76/mesh_transformer/layers.py#L303")[GPT-J-6B]
instead processes the $LN$ outputs in #emph[parallel]. That is, instead of
something like
$
  z arrow.l z + MLP (LN (z + CA (z)))
$
we instead have#footnote[This alternative layer was also used in PaLM
@chowdhery2022palm where it was claimed that this formulation is
$tilde.op 15 %$ faster due to the ability to fuse the $MLP$ and $CA$matrix
multiplies together (though this is not done in the GPT-J-6B repo
above).]
$ z arrow.l z + MLP (z) + CA (z) med . $
Note that a $LN$ instance is also removed.

=== RoPE Embeddings
<rope-embeddings>
A shortcoming of traditional embeddings
$x_(b s d) arrow.r x_(b s d) + p_(s d)$ is that they do not generalize
very well: a model trained on such embeddings with a maximum sequence
length $K$ will do very poorly when evaluated on longer sequences. RoPE
(Rotary Position Embedding) @su2022roformer and variants thereof can
extend the viable context length by more clever mechanisms with stronger
implicit biases.

RoPE and its variants can be motivated by a few natural conditions.
Given the queries and keys for an input $q_(s d) \, k_(s d)$
(suppressing batch indices), the corresponding attention scores
computation $a_(s s') (q_s \, k_(s'))$ should reasonably satisfy the
below:

+ The attention score should only depend on the position indices
  $s \, s'$ through their difference $s - s'$, i.e., through their
  relative distance to each other.

+ The score computation should still be efficient, i.e., based on
  matrix-mulitiplies.

+ The operation should preserve the scale of the intermediate
  representations and attention scores, in order to avoid issues with
  standard normalization.

These conditions suggest a very natural family of solutions: just rotate
the usual queries by some fixed element of $S O (d)$ using a generator
proportional to the position index and rotate the keys by the conjugate
element. That is, replace the $q_(s d) \, k_(s d)$ by
$
  q'_( s d )&eq.triple [e^( i s hat(n)dot T ) ]_( d d' ) q_( s d' ) eq.triple R(s)_( d d' ) q_( s d' ) \
  k'_( s d )&eq.triple [e^( -i s hat(n)dot T ) ]_( d d' ) k_( s d' ) eq.triple R(s)^( dagger )_( d d' ) k_( s d' ) ,
$<eq_rope>
which makes their dot-product is
$q'_(s d) k'_(s' d) = R (s - s ') q_(s d) k_(s d')$.

Performing the above computation with a dense element of $S O (D)$ is
infeasible, as it would require a new dense matrix-multiply by a unique
$D times D$ matrix at each sequence position#footnote[For one, the
$cal(O) ( S D ^2 )$ memory cost to store the matrices
would be prohibitive. The FLOPs cost is only $2 B S D^2$, the same as
for other matrix multiplies, but because different matrices are needed
at position (it's a batched matrix multiply), these FLOPs would be much
more GPU memory-bandwidth intensive.] In the original RoPE paper, the
rotation $hat(n)$ was chosen such that the matrices are $2 times 2$
block-diagonal with the entries of the form#footnote[If $D$ isn't even,
the vectors are padded by an extra zero.]
$
  R (s)_([d : d + 2] [d : d + 2]) & = mat(delim: "(", cos (s theta_d), - sin (s theta_d); sin (s theta_d), cos (s theta_d))
$
where $ theta_d & = 10^(- 8 d \/ D) med . $ The RoPE memory costs are thus $cal(O) ( K D
)$#footnote[A single RoPE buffer can be shared amongst all attention layers, amortizing the memory
    costs.]. The sparsity present in this constrained form of the RoPE matrices means that
@eq_rope] can be computed in $cal(O) ( B S D )$ time, rather than $cal(O) ( B S D ^2 )$, as it would
be for a general rotation matrix. See the paper for explicit expressions.

=== Flash Attention <subsec_flash_attention>
Flash Attention @dao2022flashattention@dao2023flashattention2 optimizes
the self attention computation by never materializing the
$cal(O) ( S ^2 )$ attention scores in off-chip
memory. This increases the arithmetic intensity of the computation and
reduces the activation memory required, at the expense of needing
recomputation in the backwards pass.

The central idea is to decompose the attention computation in the
following way. Dropping the batch index, let
$q_(s d) \, k_(s d) \, v_(s d)$ be the queries, keys, and values, and
$z_(s d)$ be the final output. Splitting into attention heads as in
$q_(s d) = q_(s (a h)) arrow.r q_(s a h)$ and similar, the computation
is#footnote[We omit the usual $sqrt(D \/ A)$ normalization factor inside
the Softmax to de-clutter the presentation. Really, this normalization
should just be enforced at the level of the matrices which are used to
generate the queries, keys, and values, anyway.]
$
  z_( s a h ) &= SM_( s' ) ( q_( s a h' ) k_( s' a h' ) ) v_( s' a h )
$
which is then concatenated as
$z_(s (a h)) -> z_(s d)$ to get the result. We are omitting the
(very important) causal mask for clarity of presentation. Because each
attention head computation is identical, we also omit the $a$-index
going forward in this section.

The issue is that a naive computation would compute all
$cal(O) ( S ^2 )$ components of the attention scores
$q_(s h') k_(s' h')$ for each attention head and their exponential all
at once, which incurs a penalty of shuttling back and forth
$cal(O) ( S ^2
)$ elements to and from on-chip memory multiple times in order
to get the final $z_(s h)$ outputs (in addition to being potentially
memory expensive). Flash Attention functions by instead computing the
exponentials in stages with fewer memory transfers and never populating
the attention scores or exponentials on off-chip memory.

This works by first chunking all of the inputs along their sequence
dimensions as in:

- $q_(s h) = q_((i r) h) arrow.r q_(i r h)$ where
  $i in {0 \, dots.h \, I - 1}$ and $r in {0 \, dots.h \, R - 1}$ with
  $S = R I$

- $k_(s h) = k_((j c) h) arrow.r k_(j c h) \, v_(s h) = v_((j c) h) arrow.r v_(j c h)$
  where $j in {0 \, dots.h \, J - 1}$ and $c in {0 \, dots.h \, C - 1}$
  with $S = J C$

The chunk sizes are determined by memory constraints, as discussed
below. Then, the per-attention-head computation is equivalently written
as
$
  z_(i r h) &= SM_(j c) ( q_( i r h' ) k_( j c h' ) ) v_( j c h ) \
  &= ( exp ( q_( i r h' ) k_( j c h' ) ) ) / ( sum_( j c )exp ( q_( i r h'' ) k_( j c h'' ) ) ) v_( j c h ) \
  &eq.triple (sum_( j ) Z_( i r j h ) ) / ( sum_( j'c ) exp (q_( i r h'' ) k_( j'c h'' )) ) \
  &eq.triple (sum_( j ) Z_( i r j h ) ) / (sum_( j' )L_( i j'r )) \
  &eq.triple ( Z_( i r h ) ) / (L_( i r ))
$
where we introduced the notation which will be used
in the algorithm below. The algorithm proceeds similarly to how it's
outlined above: we compute in chunks, looping over $i$ and an inner $j$
loop which is used to compute the numerator and denominator
simultaneously.

Ignoring the important causal mask and not tracking the maximum logits
(which we should do for numerical stability), the basic version which
captures the essentials of the algorithm is below. Additional
recomputation is needed for the backwards pass.

#figure(
  kind: "algorithm",
  supplement: [Algorithm],
  caption: [Flash Attention (Naive - Missing causal mask/max tracking.)],
  pseudocode-list(booktabs: true)[
    + *For* $i in ...$ #h(1fr) `# Computing outputs z[i, r, h] for all r, h`
      + Initialize off-chip tensor $z_(i r h)$ to zeros
      + Move $q_(i r h)$ on-chip, instantiate temp $Z_(i r h)$ to zeros on-chip.
      + *For* $j in ...$ #h(1fr) `# On-chip compute. r, c indices processed in parallel.`
        + Move $k_(j c h) \, v_(j c h)$ on-chip $Z_(i r h) arrow.l Z_(i r h) + exp (q_(i r h') k_(j c h')) v_(j c h)$
        + Update numerator $L_(i r) arrow.l L_(i r) + sum_c exp (q_(i r h') k_(j c h'))$
      + Update denominator $z_(i r h) arrow.l Z_(i r h) / L_(i r)$ #h(1fr) `# Write result off-chip`
  ],
) <algo_fa_fwd_basic>


We now analyze the memory transfer costs. As a baseline, vanilla
attention requires $cal(O) ( S
^2+ D S )$ memory transfers per attention head, where the two
factors come from the attention scores and $q \, k \, v$, respectively.
For flash attention, we no longer shuttle the attention scores off-chip,
but $k \, v$ are repeatedly moved back and forth. These transfers form
most of the memory operations in the inner loop above, which access
$cal(O) ( I J C H ) ~
cal(O) (  (H S ^2 )/ R  )$ elements over the
lifetime of the algorithm (per attention head). The factor $H \/ R$
determines the memory-access advantage, and this number is bound by the
on-chip memory size. The on-chip bytes from the queries, keys, and
vectors take $cal(O)
( C H + R H )$ memory and the temporaries from attention
scores and exponentials require $cal(O) ( R C )$. If we
have $M$ bytes of on-chip memory, then we have the constraint
$C H + R H + R C lt.tilde M$, and assuming assuming the chunks were
chosen to maximize on-chip memory usage, $H / R tilde.op H^2 / M$. Since
$M tilde.op 10^5$ bytes on 2023 GPUs, this is a small factor for the
typical head dimensions $H tilde.op 64$, as desired.

Flash attention is also a big win for activation memory: a naive
algorithm has a $cal(O) ( A B S
^2 )$ per-layer contribution to activation memory due to
needing to save the attention weights, but these are discarded and
re-computed for flash attention. The only additional memory cost comes
from the $cal(O) ( A B S )$ elements in the $ell_(a b s)$
statistics, which are dominated by the $cal(O) ( B S D )$
costs from needing to save inputs, and hence negligible.

==== The Details <subsubsec_fa_details>
Here we give more detailed descriptions of the flash-attention forwards
and backwards passes.

For the forwards pass, we add in maximum-logits tracking for more
numerically stable exponential computation and the causal mask. The
causal mask $C_(s s') = C_((i r) (j c))$ is zero if $s gt.eq s'$ and
$- infinity$ otherwise. The algorithm is as below.



#figure(
  kind: "algorithm",
  supplement: [Algorithm],
  caption: [Flash Attention Forward Pass],
  pseudocode-list(booktabs: true)[
    + *For* $i in ...$ `#Computing outputs  z[i, r, h] for all r, h`
      + Initialize off-chip tensors $z _( i r h ),  ell _( i r )$ to zeros #h(1fr)
      + Move $q _( i r h )$ on-chip, instantiate temp $Z _( i r h )$ to zeros and $M ^"new" _( i r ), M ^"old" _( i r )$ to $-infinity $ on-chip
      + *For* $j in ..$ #h(1fr) `# On-chip compute. r, c indices processed in parallel`
        + Move $k_( j c h ),v _( j c h )$ on-chip
        + $S_( i r j c ) <- q_( i r h' ) k_( j c h' ) + C_( i j r c )$ #h(1fr) `# logits + causal mask`
        + $M^"new"_( i r ) <- max ( M^"old"_( i r ), max_( c ) S_( i r j c ) )$
        + $Z _( i r h ) <-   Z _( i r h ) +exp  (  S _( i j r c ) - M ^"new" _( i r )   ) v _( j c h )$ #h(1fr) `# Update numerator`
        + $L_( i r ) <- e^( M^"old"_( i r ) - M^"new"_( i r ) ) L_( i r ) +sum_( c )exp ( S_( i j r c ) - M^"new"_( i r ) )$#h(1fr) `# Update denominator`
        + $M^"old"_( i r ) <- M^"new"_( i r )$
      + $z _( i r h ) <- (Z _( i r h ))/(L _( i r ))$, $ ell _( i r ) <- M ^"old" _( i r ) + ln L _( i r )$#h(1fr) Write results off-chip. $ell _( i r ) $for backwards
  ],
) <algo_fa_fwd_advanced>


For the backwards pass, the main complication comes from computing
derivatives with respect to the attention scores. Recalling the $SM$
derivative @eq_softmax_derivative.

given gradients
$(partial cal(L) ) / ( partial z_( i r j ) ) eq.triple g_( i r j )$
we have the building blocks#footnote[The fact that we can replace the
$j'$ sum with the cached attention outputs in the final derivative below
is crucial.]
$
  ( partial P_( i r j c ) ) / ( partial S_( i r j'c\' ) ) &= P_( i r j c)delta_( j j' )delta_( c c\') - P_( i j r c ) P_( i r j'c\' )\
  ( partial cal(L) ) / ( partial P_( i r j c ) ) &= g_( i r h ) v_( j c h )\
  ( partial cal(L) ) / ( partial S_( i r j c ) ) &= g_( i r h ) ( partial P_( i r j c ) ) / ( partial S_( i r j'c\' ) )\
  &= g_( i r h ) ( P_( i r j c ) v_( j c h ) - P_( i r j c ) P_( i r j'c\' ) v_( j'c\'h ) )\
  &= g_( i r h ) ( P_( i r j c ) v_( j c h ) - P_( i r j c ) z_(i r h ) )\
  &= P_( i r j c )(( partial cal(L) ) / ( partial P_( i r j c ) ) -g_( i r h ) z_(i r h ) )
$<eq_fa2_derivative_building_blocks>
from which we compute
$
  ( partial cal(L) ) / ( partial v_( j c h ) ) &= g_( i r h ) P_( i r j c )\
  ( partial cal(L) ) / ( partial q_( i r h ) ) &= ( partial cal(L) ) / ( partial S_( i r j c ) )k_( j c h )\
  ( partial cal(L) ) / ( partial k_( j c h ) ) &= ( partial cal(L) ) / ( partial S_( i r j c ) )q_( i r h )
$ Above we let
$P_( i j r c ) eq.triple SM_( j c) ( S_( i r j c ) )$
where $S_(i j r c) equiv q_(i r h') k_(j c h') + C_(i j r c)$, keeping
notation similar to the above algorithm. All of this suggests a very
similar algorithm to the above. Using the unfortunate, but common,
notation, $dif X = ( partial cal(L) )/( partial X )$ the
algorithm is#footnote[In the FA2 paper, they actually pre-compute the
$g_(i r h) z_(i r h)$ sum prior to the main loop and store it in a
tensor they call $D$. And in the official `triton` example, $dif q$
is computed in a separate loop. So, take the below as more of a
guideline than a strict recipe.]:



#figure(
  kind: "algorithm",
  supplement: [Algorithm],
  caption: [Flash Attention Backwards Pass],
  pseudocode-list(booktabs: true)[
    + *For* $i in ...$
      + Initialize off-chip tensors $dif q _( i r h ), dif k _( j c h ), dif v _( j c h )$ to zeros
      + Move $z_(i r h) \, q_(i r h) \, g_(i r h)$ and the cached $ell_(i r)$ + on-chip.
      + *For* $j in ...$ #h(1fr) `# On-chip compute. r, c processed in parallel`
        + Instantiate $P _( i r j c ),dif P _( i r j c ) ,dif S _( i r j c )$ to zeros.
        + Move $k_(j c h) \, v_(j c h)$ on-chip
        + $P_(i r j c) arrow.l exp (q_(i r h') k_(j c h') + C_(i j r c) + ell_(i r))$ #h(1fr) `# Get probabilities`
        + $dif P_( i r j c ) <- g_( i r h) v_( j c h )$ #h(1fr) `# Get derivatives w.r.t. P`
        + $dif S _( i r j c ) <- P _( i j r c ) ( dif P _( i j r c ) - g _( i r h ) z _( i r h )  )$ #h(1fr) `# Get derivatives w.r.t. S`
        + $dif k _( j c h ) <- dif S _( i r j c ) q _( i r h )$ #h(1fr) `# Get derivatives w.r.t. q, k, v`
        + $dif v_( j c h ) <- g_( i r h )P_( i r j c )$
        + Write $dif k, dif v$ derivatives to off-chip
        + $dif q_( i r h ) <- dif q_( i r h )+ dif S_( i r j c ) k_( j c h )$
    + Write $dif q$ derivative to off-chip
  ],
) <algo_fa_bwd_advanced>


=== Linear Attention <subsec_linear_attn>
Linear attention @katharopoulos2020transformersrnnsfastautoregressive
removes the $SM$ operation in the attention layer in order to reduce
the inference costs in terms of compute and time, both.

To review, the (single-head) attention operation is $
z _( s d ) &= SM_( s\' ) ( q _( s d\' ) k _( s\'d\' ) ) v _( s\'d )\
&eq.triple A _( s s\' ) v _( s\'d )  .
$ In order to generate the final, $s = - 1$ token using
a kv-cache, generation time then requires reading in
$cal(O) ( D S )$ bytes and performing
$cal(O) ( D S )$ operations to generate the new token.
However, if we remove the $SM$, then we can write the above as
$
  z_( s d ) &= q_( s d\' ) k_( s\'d\' ) v_( s\'d )\
  &= q_( s d\' )B_(d\' d ) .
$ This would let us cache the
$cal(O) ( D ^2 )$ $B_(d' d)$ matrix and generating
the next token only takes $cal(O) ( D ^2 )$
operations, an $cal(O) ( D/S
)$ improvement on both fronts.

The essential point is that for standard attention, the entire
$A_(s s')$ matrix must be computed anew for each new token, while
$B_(d d')$ can instead be iteratively updated via a cheap computation.

The causal masking looks a little different for linear attention. The
causal mask is not needed during vanilla next-token generation, but is
for parallelized training. Computing of the $z_(s d)$ in parallel, as in
training, requires generating $S$ different matrices $B_(d' d)^s$, one
for each token position:
$B_(d' d)^s = CUMSUM_s (k_(s d') v_(s d))$, effectively.
Flash-attention-like techniques can be used to avoid materializing all
of the $cal(O) ( S D ^2 )$ elements at once.

= State Space Models
<state-space-models>
== Intro<sec_ssm_intro>


The all-to-all attention mechanism of transformers is a pain: $cal(O)( S^( 2 ) )$ compute at
training time and $cal(O)( S )$ next-token generation. State space models return, more or less, to
the old LSTM type strategy of encoding the conditional history into finite-sized state. The dream is
faster generation and better memory efficiency:
- Parallelizable#footnote[Better parallelization support is what differentiated S4 models from their
  RNN/LSTM predecessors; see @rnns_and_ssm.] $cal(O)( S )$ training.
- Constant $cal(O)( 1 )$ generation.
- Sequence-length-independent state, reducing the inference-time memory bandwidth burden compared to the kv-cache; @sec_kv_cache.


== S4 <sec_s4>
The S4 model of @s4 is a good starting point. These are based off a
continuous representation in which some input signal#footnote[We use the
notation of the mamba paper @mamba, which differs from that of the S4
paper @s4.] $x_a (t)$ is converted to an output $y_c (t)$ via an in
intermediate latent variable $h_b (t)$, with the above related as in
$
  partial_( t )h_( b )(t) &= A_( b b\' )h_( b\' )(t) + B_( b a )x_( a )(t)\
  y_( c )(t) &= C_( h b )h_( b )(t) + D_( c a )x_( a )(t) .
$<eq_s4_continuous>
The capitalized tensors are the learnable weight
matrices. $D$ is often set to zero in the literature. Basically, the
information in the sequence $x_s$ is stored in $h_s$, an internal memory
for the model, much like the RNN/LSTM models of the past.

For discrete sequences, we discretize: $
h _( b s ) &= A _( b b\' )h _( b\' (s-1) ) + B _( b a )x _( a s )\
y _( c s ) &= C _( c b )h _( b s ) + D _( c a )x _( a s )   .
$<eq_s4_discrete> where one can also relate these weights to those in
@eq_s4_continuous given the
discretization scheme (see @sec_mamba).

Subject to the initial condition $h_b^(- 1) = 0$, the above solves to
$
  y_s & = sum_(s' = 0)^s C dot.op A^(s - s) dot.op B dot.op x_(s') + D x_s med \,
$
omitting hidden dimension indices. Proper normalization of the various
weights is non-trivial; see @s4 for details. Further, diagonalization
clearly makes the $A^(s - n)$ computation easier, but care must be taken
here, too. Clearly, the above computation is highly parallelizable. The
/* S4 (and mamba) papers describe @eq_s4_soln as */
a

Writing the above operation as $y_(c s) = Sigma_(c a s s') x_(a s')$,
one can build an non-linear S4 layer by acting on the output with a
non-linearity and then mixing feature dimensions with a weight matrix:
$ z_(c s) & = W_(c c\') phi (Sigma^(c\' a s s') x_(a s')) $ Assuming
the $c$ and $a$ hidden dimensions have the same size, the operations can
then be naturally composed.

Taking all hidden dimensions to have size $cal(O) ( D )$, the number of learnable weights is $cal(O)
( D ^2 )$. Training can be parallelized across the sequence dimension (via the representation
/* @eq_s4_soln;, scaling linearly in sequence */
length. Iterative generation from $x_(a s) arrow.r y_(c s)$, given knowledge of the previous hidden
state $h_(b (s - 1))$ takes only $cal(O) ( D ^2 )$ (via the representation @eq_s4_discrete;). There
is no sequence-length dependence for next-output generation, unlike for transformers, which is the
main draw here: constant-time generation.

== Mamba<sec_mamba>
A large limitation of the S4 model
@eq_s4_discrete is that the various
weights are fixed quantities which do not adjust to the
input#footnote[For instance, we could ask our architecture to process
two independent sequences concatenated together with a special separator
token in the middle. The hidden state should be reset at the separator
token and the mamba architecture would be (in-principle) capable of
this, while the S4 would not.] $x_(s d)$. Mamba @mamba extends S4 by
replacing the fixed weights by functions of the inputs. This destroys
the recursive structure and requires various techniques for an efficient
GPU implementation, which is the primary focus of the paper.

The mamba architecture is as follows, based on the implementation in
#link("https://github.com/alxndrTL/mamba.py")[`mamba.py`] and
#link("https://github.com/state-spaces/mamba")[`mamba_ssm`]. Notation
for dimensions and tensors:

- Mamba maps sequences to sequences, the same as for transformers.
  $z_(s d) = mono("mamba") (x_(s d))$. Batch dimension suppressed
  throughout.

- Various dimensions:

  - $d in (0 \, dots.h \, D - 1)$: the input's hidden dimensions,
    `d_model`.

  - $e in (0 \, dots.h \, E times D - 1)$: expanded internal hidden
    dimension. Usually $E = 2$ in practice.

  - $s in (0 \, dots.h \, S - 1)$: sequence length.

  - $n in (0 \, dots.h \, N - 1)$: another internal hidden dimension,
    controlling the size of the internal memory; `d_state`. Defaults to
    16.

  - $r in (0 \, dots.h \, R - 1)$: another internal hidden dimension,
    `d_rank`. Defaults to $ceil.l D \/ 16 ceil.r$.

  - $c in (0 \, dots.h \, C - 1)$: convolution kernel size; `d_conv`, 4
    by default. Used to convolve over the sequence dimension.

- Learnable parameters#footnote[In practice, many of these are fused
  together for more efficient matmuls. We also omit potential bias
  terms.]:

  - Two in-projectors from `d_model ` to the expanded dimension:
    $W_(e d)^(I_0)$, $W_(e d)^(I_1)$.

  - Out-projector from the expanded internal dimension back to
    `d_model ` $W_(d e)^O$.

  - Two projectors used in creating the intermediate $Delta_(s e)$:
    $W_(r e)^(Delta_0)$, $W_(e r)^(Delta_1)$.

  - Projectors for creating the intermediates $B_(s n)$ and $C_(s n)$:
    $W_(n e)^B$, $W_(n e)^C$

  - Convolutional kernel $W_(e c)^K$.

  - Selective-scan weights $W_(e n)^A$.

  - Residual connection weights $W_e^D$.

The notation here is not the same as that of the papers. We write all
learnable weights as $W_dots.h^X$.

Mamba blocks then perform the following logical operation:

#figure(
  kind: "algorithm",
  supplement: [Algorithm],
  caption: [Mamba],
  pseudocode-list(booktabs: true)[
    + *Inputs*: tensor $x_(s d) in bb(R)^(S times D)$
    + $x_(s e)^0 = W_(e d)^(I_0) x_(s d)$, $x_(s e)^1 = W_(e d)^(I_1) x_(s d)$ #h(1fr) `# Create expanded tensors from inputs (can fuse)`
    + $x_(s e)^2 = K_(e s s') star.op x_(s e)^1$ #h(1fr) `# Grouped conv. over the seq dim using W^K.`
    + $x_(s e)^3 = phi (x_(s e)^2)$ #h(1fr) `# Elementwise non-linearity (silu)`
    + $x_(s e)^4 = mono("selective_scan") (x_(s e)^3)$ #h(1fr) `# Selective scan`
    + $x_(s e)^5 = x_(s e)^4 times.circle phi (x_(s e)^0)$ #h(1fr) `# Elementwise product and non-linearity (silu)`
    + *Return* $z_(s d) = W_(d e)^O x_(s e)^5$ #h(1fr) `#Project back down.`
  ],
)<algo_mamba_1>
where `selective_scan` operation is the above is#footnote[The
`mamba_ssm` and `mamba.py` implementations differ in the first step in
that the latter optionally applies a norm operator post-projection. The
exponentials here might seem odd, but are probably motivated by the
existence of good cumulative sum kernels, which is how the exponents can
be computed.]



#figure(
  kind: "algorithm",
  supplement: [Algorithm],
  caption: [Selective Scan],
  pseudocode-list(booktabs: true)[
    + *Inputs*: $x_(s e) in bb(R)^(S times E)$
    + $B_(s n) = W_(n e)^B x_(s e)$ #h(1fr) `# Create intermediates B, C, Delta (can fuse).`
    + $C_(s n) = W_e^C x_(s e)$
    + $Delta_(s e) = W_(e r)^(Delta_1) W_(r e)^(Delta_0) x_(s e)$.
    + Solve recursion, subject to $h_((- 1) e n) = 0$:
    - $h_( s e n ) &= exp ( Delta_( s e ) W^(A)_( e n ) ) h_( (s-1)e n) + Delta_( s e )B_( s n )x_( s e )$
    - $y_( s e ) &= C_( s n )h_( s e n ) + W^( D )_( e )x_( s e )$
    - $=> y_( s e ) &= C_( s n ) (sum_( s\'=0 )^( s )e^( Delta_( s e )W^(A)_( e n ) ) times ... times e^( Delta_( (s\'+1)e )W^(A)_( e n ) ) Delta_( s\'e ) B_( s\'n ) x_( s\'e ) ) + W^( D )_( e ) x_( s e )$
    - $=> y_( s e )&= C_( s n ) (sum_( s\'=0 )^( s )product_( s\'\'=s\'+1 )^( s )e^( Delta_( s\'\'e
          )W^(A)_( e n ) ) Delta_( s\'e ) B_( s\'n ) x_( s\'e ) ) + W^( D )_( e ) x_( s e)$
    + *Return* $y_(s e) in bb(R)^(S times E)$
  ],
)<algo_mamba1_scan>


As noted above, the creation of the intermediates
$x_(s e)^0 \, x_(s e)^1 \, B_(s n) \, C_(s n)$ and part of $Delta_(s e)$
can all be formed in a single large matmul.

== Mamba 2
<mamba-2>
Mamba2 introduces some changes:

- The $n$-dimension is expanded to `ngroups` such dimensions (though
  `ngroups`=1 is the default), with associated index
  $g in (0 \, dots.h \, G - 1)$, $G equiv mono("ngroups")$. Adding a
  non-trivial `ngroups` seems completely degenerate with expanding the
  $n$ dimension of size `d_state` to size
  $mono("d_state") times mono("ngroups")$.

- A head-index $a in (0 \, dots.h \, A - 1)$ ($A equiv mono("nheads")$)
  and head dimension $h in (0 \, dots.h \, H)$ ($A times H = E$) are
  introduced, analogously to transformers.

- The $e$-index from two selective-scan weights is removed: they are now
  per-head scalars $W_a^A \, W_a^D$.

- The intermediate $Delta_(s a)$ is also reduced to a per-head,
  per-sequence-position scalar, with respect to the hidden dimension.
  This tensor is now created via a single matmul with weight
  $W_(a e)^Delta$.

- The short 1D convolution is now also taken over the $B$ and $C$
  intermediates with kernels $W_(g n c)^(K_B)$, $W_(g n c)^(K_B)$.

The updated model:



#figure(
  kind: "algorithm",
  supplement: [Algorithm],
  caption: [Mamba2],
  pseudocode-list(booktabs: true)[
    + *Inputs* $x_(s d) in bb(R)^(S times D)$
    + $x_(s e)^0 = W_(e d)^(I_0) x_(s d)$, $x_(s e)^1 = W_(e d)^(I_1) x_(s d)$ #h(1fr) `# Create expanded tensors from inputs (can fuse)`
    + $x_(s e)^2 = K_(e s s') star.op x_(s e)^1$ #h(1fr) `# 1D grouped convolution over the sequence dimension (fused)`
    + $x_(s e)^3 = phi (x_(s e)^2)$ #h(1fr) `# Elementwise non-linearity (silu)`
    + $x_(s e)^4 = mono("selective_scan2") (x_(s e)^3)$ #h(1fr) `# Selective scan`
    + $x^5_( s e ) = NORM (x^4_( s e ) times.circle phi ( x^0_( s e ) ) )_( e )$ #h(1fr) `# Elementwise product, non-linearity, and norm (RMS)`
    + *Return* $z_(s d) = W_(d e)^O x_(s e)^5$ #h(1fr) `# Project back down.`
  ],
)<algo_mamba_2>



The mechanical differences are the normalization step and the details of
the `selective_scan2` operation, which is essentially the same as
before, but now the hidden $e$ is split into multiple attention heads,
analogously to transformer models:



#figure(
  kind: "algorithm",
  supplement: [Algorithm],
  caption: [Selective Scan 2: `selective_scan2`],
  pseudocode-list(booktabs: true)[
    + *Inputs*: $x_(s e) in bb(R)^(S times E)$ $x_(s a h) = x_(s (a h)) = x_(s e)$ #h(1fr) `# Break the inputs up into attention heads.`
    + $B_(s g n) = W_(g n e)^B x_(s e)$ #h(1fr) `# Create intermediates B, C, Delta (can fuse)`#footnote[The `mamba_ssm` and `mamba.py` implementations differ here in that the latter optionally applies a norm operator post-projection.].
    + $C_(s g n) = W_(g n e)^C x_(s e)$ $Delta_(s a) = W_(a e)^Delta x_(s e)$.
    + $Delta_(s a) = mono("Softplus") (Delta_(s a))$. #h(1fr) `# For some reason.` $mono("Softplus") (x) equiv ln (1 + e^x)$.
    + $B_(s g n) = K_(g n s s')^B star.op B_(s g n)$ #h(1fr) `# 1D grouped conv. over the seq. dim. (fused)`
    + $C_(s g n) = K_(g n s s')^C star.op C_(s g n)$
    + Solve recursion, subject to $h_((- 1) g a h n) = 0$:
    - $h_( s g a h n ) &= exp ( Delta_( s a ) W^(A)_( a ) ) h_( (s-1)g a h n) + Delta_( s a )B_( s g n )x_( s a h )$
    - $y_( s a h ) &= C_( s g n )h_( s g a h n ) + W^( D )_( a )x_( s a h )$
    - $=> y_( s a h ) &= C_( s g n ) (sum_( s\'=0 )^( s )e^( Delta_( s a )W^(A)_( a ) ) times ... times e^( Delta_( (s\'+1)a )W^(A)_( a ) ) Delta_( s\'a )B_( s\'g n ) x_( s\'a h ) ) + W^( D )_( a ) x_( s a h )$
    - $=> y_( s a h )&= C_( s g n ) (sum_( s\'=0 )^( s )product_( s\'\'=s\'+1 )^( s )e^( Delta_( s\'\'a )W^(A)_( a ) ) Delta_( s\'a )B_( s\'g n ) x_( s\'a h ) ) + W^( D )_( a ) x_( s a h)$
    + *Return* $y_(s e) = y_(s (a h))$ #h(1fr) `# Concatenate the heads back together. `
  ],
)<algo_mamba2_scan>

As before, many of the matmuls can be performed as one big operation,
and the three short convolutions can be similarly fused into a single
convolution. The two algorithms ~@algo_mamba1_scan and
~@algo_mamba2_scan are nearly identical; they just differ in some
tensor shapes.

=== Mamba2 Duality with Attention
<mamba2-duality-with-attention>
There are only two steps in which tokens at different temporal positions
interact in the Mamba2 model:

+ In the short 1D convolution.

+ In the recurrence relation, where we create the intermediate
  $
    z_(s a h) & = C_(s g n) (sum_(s' = 0)^s e^(Delta_(s a) W_a^A) times dots.h times e^(Delta_((s ' + 1) a) W_a^A) Delta_(s' a) B_(s' g n) x_(s' a h)) equiv M_(a s s') x_(s' a h)
  $
  which is the most complicated step of the model.

As noted above, the second case is ultimately just a matrix-multiply on
the input tensors $x_(s a h)$ with the tensor $M_(a s s')$, where
operations across attention head are all independent. The $M_(a s s')$
tensor has $cal(O) ( A S ^2 )$ elements, which we
clearly do not want to concurrently materalize. All of this should sound
familiar: the above is exactly analogous to the structure and problems
of flash attention, @subsec_flash_attention, the only difference
begin how the linear operator $M_(a s s')$ is constructed:
$M_( a s s\' ) = SM (q^( a )_( s h )k^( a )_( s\'h ) )$
in standard attention, and as above for Mamba2, say. This is the
“duality\" discussed in @dao2024transformersssmsgeneralizedmodels and
the broad strokes of the efficient algorithm implementation for Mamba2
echos that of flash attention: partition the computation over the
sequence dimensions and compute $z_(s a h)$ in chunks over the
$s$-dimension, so as to avoid realizing any
$cal(O) ( S ^2 )$ tensors.

Similar statements hold for the original Mamba; the index names and
choices just make the analogy more readily recognizable in Mamba2.

=== Details: Cumsums, Chunking, and the Mamba2 Scan
<details-cumsums-chunking-and-the-mamba2-scan>
Some more details about how to compute the recursion solution in
@algo_mamba2_scan. Similarly to the previous section, let
$
  z_( s a h ) &= C_( s g n ) (sum_( s\'=0 )^( s )e^( Delta_( s a )W^(A)_( a ) ) times ... times e^( Delta_( (s\'+1)a )W^(A)_( a ) ) Delta_( s\'a )B_( s\'g n ) x_( s\'a h ) )\
  &eq.triple C_( s g n )cal(A)_( s s\'a )Delta_( s\'a )B_( s\'g n ) x_( s\'a h )\
  &eq.triple C_( s g n )cal(A)_( s s\'a )cal(B)_( s\'g a n h )
$ The above is the most complex part of the Mamba2
model and the official `mamba_ssm` repo takes a multi-step approach to
its computation. Two primary points:

+ The matrix $cal(A)_(s s' a)$ vanishes for $s' > s$ (causality).

+ As in flash attention, we wish to chunk over the sequence dimension to
  avoid every realizing the full $cal(A)_(s s' a)$ tensor.

The chunked version is then
$
  z_(c l a h) & = C_(c l g n) cal(A)_(c c\' l l' a) cal(B)_(c\' l' g a n h) med \,
$
where we have chunked the $a$-index into the $c \, l$ pair (with $c$
indexing the chunk). The chunked computation breaks down into two
further cases, based on the values of the $c \, c\'$ indices#footnote[The
$c\' > c$ cases are trivial as $cal(A)_(c c\' l l' a)$ vanishes.]:

- $c = c\'$: these cases are effectively smaller versions of the entire,
  unchunked computation, and hence shares in its sparsity in that
  $cal(A)_(c c\' l l' a)$ vanishes for $l' > l$.

- $c > c\'$: there is no sparsity here, as $cal(A)_(c c\' l l' a)$ will be
  generically non-zero for all $l \, l'$.

==== The $c = c\'$ Cases
<the-cc-cases>
Logically, we compute the scan using cumulative sums and
matrix-multiplies. Let
$Delta_(s a) W_a^A = A_(s a) = A_((c l) a) = A_(c l a)$ so that
$
  cal(A)_( s s\'a ) &=
  cases(
  e^( A_( s a )) times ... times e^( A_( (s\'+1)a )) =exp ( sum_( s\'\'=s\'+1 )^( s )A_( s\'\'a ) ) wide & s >= s\',
  0 & s \< s\'
  )\
  &eq.triple e^( bold(A)_( s s\'a ) ) .
$ Sharding and taking only the diagonal terms, the
above turns into (no sum over the repeated $c$-index):
$
  cal(A)_( c c\' l l\'a ) &=
  cases(
  e^( A_( c l a )) times ... times e^( A_( c(l\'+1)a )) =exp ( sum_( l\'\'=l\'+1 )^( l )A_( c l\'\'a ) ) wide & l >= l\' ,
  0 & l \< l\'
  )\
  &eq.triple e^( bold(A)_( c c\' l l\'a ) ) .
$

The argument $sans(A)_(c c\' l l' a)$ can be constructed in various ways#footnote[$CUMSUM_s X_s
  equiv sum_(s' = 0)^s X_(s')$ and $SEGSUM$ stands for “segment sum\".]:
$
  bold(A)_( c c\' l l\'a )&= SEGSUM_( l l' ) ( A_( c l a ) ) + M_( l l\' )\
  &eq.triple CUMSUM_( l )A_( c l a ) - CUMSUM_( l\' )A_( c l\'a ) + M_( l l\' ) \
  &= CUMSUM_( l ) ( A_( c l a )Z_( l l\' ) ) + M_( l l\' )\
  Z_( l l\' ) &eq.triple cases(
0 wide & l <= l\',
1 & l \> l\'
) , \
  M_( l l\' )& eq.triple cases(
-infinity wide & l < l\',
0 & l >= l\'
) ,
$<app_eq_mamba2_diag_propagator>
where the final form with the additional mask
$Z_(l l')$ is better behaved numerically, as it does not rely on
cancellations between sums. Careful attention should be paid to the
inequality symbols in the masks. The remainder of these diagonal
computations is straightforward.




==== The $c > c\'$ Cases
<the-cc-cases-1>
Now we compute the remaining off-diagonal terms. Compute one
$(c \, c ')$ chunk at a time, i.e. we compute
$
  z_(c l a h) = C_(c l g n) cal(A)_(c c\' l l' a) cal(B)_(c\' l' g a n h) med \,
$
by iterating over the $c\'$ sum, similarly to flash attention.

$cal(A)_(c c\' l l' a)$ is made up of $tilde.op e^(A_(s a))$ factors
which each serve to propagate $cal(B)_((s - 1) g a n h)$ forward one
step in time. Specifically, $cal(A)_(c c\' l l' a)$ contains all the
factors needed to get from the times specified by the $(c ' \, l ')$
indices up to the $(c \, l)$ indices: $
cal(A)_( c c\'l l\'a ) &=exp  ( sum_( s= c\'L + l\' + 1 )^( c L + l )A _( s a )  )  , quad "for" c\>c\'  .
$ We will break the above into three
factors#footnote[In the nomenclature of
@dao2024transformersssmsgeneralizedmodels, whose authors also refer to
these as the B, A, and C blocks, respectively (though we actually differ
slightly in detail from what the paper and `mamba-ssm` do).]:

+ A right-factor which propagates the $cal(B)_(c\' l' g a n h)$ from
  their disparate times $(c ' \, l ')$ all up to a common point in time.

+ A center-factor which propagates the previous element together for a
  period.

+ A left-factor which finally propagates these elements up to their
  final time slices $(c \, l)$ specified by $C_(c l g n)$.

The specific form of the factors comes from the following (non-unique)
decomposition of the preceding expression: $
cal(A)_( c c\'l l\'a )|_( c\>c\' ) &=exp  ( sum_( s= c\'L + l\' + 1 )^( c L + l )A _( s a )  ) \
&= exp  (sum _( l\'\'=0 )^( l )A_( c l\'\'a) +sum_( c\'\'=c\'+1 )^( c-1 )sum_( l=0 )^( L-1 )A_( c\'\'l a ) + sum _( l\'\'=l\'+1 )^( L-1 )A_( c\'l\'\'a)  )\
&= exp  (sum _( l\'\'=0 )^( l )A_( c l\'\'a) ) exp  (sum_( c\'\'=c\'+1 )^( c-1 )sum_( l=0 )^( L-1 )A_( c\'\'l a )  ) exp  ( sum _( l\'\'=l\'+1 )^( L-1 )A_( c\'l\'\'a)  ) \
&eq.triple U_( c l a )bold(A)_( c c\'a )T_( c\'l\'a ) ,
$ such that we have $
z_( c l c\'a h )&= C _( c l g n )U_( c l a )bold(A)_( c c\'a )T_( c\'l\'a )cal(B)_( c\'l\'g a n h )\
&= bold(C)_( c l g n a )bold(A)_( c c\'a )bold(B)_( c\'g a n h ) ,
$ which (I believe) are the C, A, and B blocks from
@dao2024transformersssmsgeneralizedmodels.

These factors can be conveniently, and succinctly, vectorized as
in#footnote[These can also be written in a form similar to
@app_eq_mamba2_diag_propagator
where we use masks instead of relying in numerically unstable
cancellations. $T_(c\' l' a) = exp (mono("sum")_l Z_(l l') A_(c\' l a))$,
$U_(c l a) = exp (mono("sum")_(l') (1 - Z_(l' l)) A_(c\' l' a))$ with
$Z_(l l')$ the mask in
@app_eq_mamba2_diag_propagator;.]:
$
  T_( c\'l\'a ) &=exp ( SUM_( l\' ) ( A_( c\'l\'a ) ) - CUMSUM_( l\' )A_( c l\'a ) )\
  bold(A)_( c c\'a )&=exp ( SEGSUM_( c c\' )A_( c a ) - A_( c\' a ) ) quad "where" quad A_( c a )eq.triple SUM_( l )A_( c l a )\
  U_( c l a ) &=exp ( CUMSUM_( l ) ( A_( c l a ) ) ) .
$

The full solution decomposed in this way is then: $
z _( c l a h ) &= C _( c l g n )e^( bold(A)_( c c\' l l\'a ) )cal(B)_( c l\'g a n h ) + M_( c c\' )C _( c l g n )U_( c l a )bold(A)_( c c\'a )T_( c\'l\'a )cal(B)_( c l\'g a n h )\
M_( c c\' ) & = cases(
1 wide & c \> c\',
0 & c <= c\'
)  .
$


A crucial computational point is that the matrix $bold(A)_(c c\'a)$ is low-rank:
$
  bold(A)_(c c\'a) &= exp ( SEGSUM_( c c\' )A_( c a ) - A_( c\' a ) )\
  & = exp (CUMSUM_( c ) A_( c a )) times exp (CUMSUM_( c\' ) A_( c\' a ) - A_(c\' a )) .
$
This means that the sum $bold(A)_( c c\'a )T_( c\'l\'a )$ can be performed in $cal(O)(S)$ time by
performing the $exp(CUMSUM_( c\' ) A_( c\' a ) - A_(c\' a )) T_( c\'l\'a )$ sum over $c\'$ first, and then multiplying by the
remaining factor. Otherwise, this decomposition wouldn't realize the optimal $cal(O)(S)$ scan
scaling.

=== Aren't These Just RNNs?<rnns_and_ssm>
Yes, but very special ones with the important computational difference that the recursion relations
are #emph[linear] in the hidden state $h$. This crucial difference makes it possible to parallelize
the operations during training. Compare @eq_s4_discrete to what typical RNN recursion relations
would look like:
$
  h_( b s ) &= phi (A_( b b\' )h_( b\' (s-1) ) + B_( b a )x_( a s ) )\
  y_( c s ) &= phi (C_( c b )h_( b s ) + D_( c a )x_( a s ) ) .
$<eq_rnn_comparison>
for some non-linearity $phi$. The recursion relations would solve to an expression with nested $phi$
factors which would make the computation of $h_(b s)$ non-associative. But in the linear $phi (x) =
x$ limit, the operations are #emph[associative] which makes them #emph[parallelizable], via known
scan algorithms @prefixSumsBlelloch.

= Training
<training>
== Memory <sec_memory_training>
In this section we summarize the train-time memory costs of Transformers
under various training strategies#footnote[A nice related blog post is
#link("https://blog.eleuther.ai/transformer-math/")[here].
<foot_eleuther_math_101>].

The memory cost is much more than simply the cost of the model
parameters. Significant factors include:

- Optimizer states, like those of

- Mixed precision training costs, due to keeping multiple model copies.

- Gradients

- Activation memory#footnote[Activations refers to any intermediate
  value which needs to be cached in order to compute backpropagation. We
  will be conservative and assume that the inputs of all operations need
  to be stored, though in practice gradient checkpointing and
  recomputation allow one to trade caching for redundant compute. In
  particular, flash attention @dao2022flashattention makes use of this
  strategy.] , needed for backpropagation.

Because the activation counting is a little more involved, it is in its
own section.

#block[
  Essentials Memory costs count the elements of all tensors in some
  fashion, both from model parameters and intermediate representations.
  The gradient and optimizer state costs scale with the former quantity:
  $cal(O)  ( N _"params"  )  cal(O)  ( L D ^2  )$,
  only counting the dominant contributions from weight matrices.
  Activation memory scales with the latter, which for a -shaped input
  gives $cal(O)  ( B D L S  )$ contributions from tensors
  which preserve the input shape, as well as
  $cal(O) ( A B L S ^2 )$ factors from attention matrices.

]
=== No Sharding
<no-sharding>
Start with the simplest case where there is no sharding of the model
states. Handling the different parallelism strategies later will be
relatively straightforward, as it involves inserting just a few factors
here and there.

==== Parameters, Gradients, Optimizer States, and Mixed Precision <sec_params_grads_optim_mem>

Memory from the bare parameter cost, gradients, and optimizer states are
fixed costs independent of batch size and sequence-length (unlike
activation memory), so we discuss them all together here. The parameter
and optimizer costs are also sensitive to whether or not mixed-precision
is used, hence we also address that topic, briefly. We will assume the
use of #footnote[Which stores
#link("https://pytorch.org/docs/stable/generated/torch.optim.Adam.html")[two different running averages]
per-model parameter.] throughout, for simplicity and concreteness. It
will some times be useful below to let $p$ to denote the precision in
bytes that any given element is stored in, so corresponds to $p = 4$,
for instance. Ultimately, we primarily consider vanilla training in
$p = 4$ precision and / ($p = 4$/ $p = 2$) mixed-precision, other,
increasingly popular variants to exist, so we keep the precision
variable where we can.

Without mixed precision, the total cost of the ($p = 4$ bytes) model and
optimizer states in bytes is then $
M _"model" & = 4 N _"params"  , quad M_"optim" = 8 N _"params"
quad ( "no  mixed  precision", ) p=4)
$<eq_optimizer_states_mem_no_mp> where, from the previous section, the pure
parameter-count of the decoder-only Transformers architecture is
$
  N_"params" & approx (4 + 2E) L D^2 times ( 1 + cal(O) ( V / ( D L ) ) + cal(O) ( 1 / D ) ) .
$<eq_approx_params_no_sharding> where the first term comes from the weight
matrices#footnote[So, in the usual $E = 4$ case, the layers are twice as
costly as the layers.], the first omitted subleading correction term is
the embedding matrix, and the last comes from biases, instances, and
other negligible factors. The optimizer states cost double the model
itself.

The situation is more complicated when mixed-precision is used
@micikevicius2018mixed. The pertinent components of
mixed-precision#footnote[A note on the implementation of mixed-precision
in : usually mixed-precision occurs by wrapping the forward pass in a
context manager, . The default behavior is to then create copies of some
tensors in lower-precision and do the forward pass with those. For
instance, this is done with matrix-multiplies whose arguments and
outputs will be in , but for sums the inputs and outputs will all be ,
for vanilla mixed-precision usage. Consequently, any such versions of
tensor will often persist effectively as contributors to activation
memory, since the backwards pass will need those same tensors. This can
be verified by inspecting the saved tensors: if is the output of a
matrix-multiply in such an autocast context, will be a copy of the
weights used to perform the matrix-multiply. In effect, the cost of the
model weights which are used for the actual forward pass are only
materialized within the lifetime of the context manager.]:

- A half-precision ($p = 2$ bytes) copy of the model is used to perform
  the forwards and backwards passes

- A second, \"master copy\" of the model is also kept with weights in
  full $p = 4$ precision

- The internal states are kept in full-precision

Confusingly, the master copy weights are usually accounted for as part
of the optimizer state, in which case the above is altered to
$
  M_"model" & = 2 N_"params" , quad M_"optim" = 12 N_"params"
  quad ("mixed precision") .
$<eq_optimizer_states_mem_mp>
The optimizer state is now six times the cost of the actual model used to process data and the costs
of @eq_optimizer_states_mem_mp are more than those of @eq_optimizer_states_mem_no_mp;. However, as
we will see, the reduced cost of activation memory can offset these increased costs, and we get the
added benefit of increased speed due to specialized hardware. The above also demonstrates why
training is so much more expensive than inference.

==== Gradients
<gradients>
Gradients are pretty simple and always cost the same regardless of
whether or not mixed-precision is used: $
M_"grad" & = 4 N _"params"   .
$<eq_grad_memory> In mixed precision, even though the gradients are
initially computed in $p = 2$, they
#link("https://huggingface.co/docs/transformers/v4.20.1/e n/perf_train_gpu_one#anatomy-of-models-memory")[have to be converted]
to $p = 4$ to be applied to the master weights of the same precision.

==== Activations
<activations>
Activations will require a more extended analysis
@korthikanti2022reducing. Unlike the above results, the activation
memory will depend on both the batch size and input sequence length, $B$
and $S$, scaling linearly with both.

===== Attention Activations
<attention-activations>
We will count the number of input elements which need to be cached. Our
\-shaped inputs to the attention layer with $B D S$ elements are first
converted to $3 B D S$ total query, key, value elements, and the
query-key dot products produce $A B S^2$ more, which are softmaxed into
$A B S^2$ normalized scores. The re-weighted inputs to the final linear
layer also have $B D S$ elements, bringing the running sum to
$B S (5 D + 2 A S)$

Finally, there are also the dropout layers applied to the normalized
attention scores and the final output whose masks must be cached in
order to backpropagate. In torch, the mask is a tensor, but
#link("https://github.com/pytorch/pytorch/issues/41571")[surprisingly]
these use one #emph[byte] of memory per element, rather than one bit
#footnote[As you can verify via]. Given this, the total memory cost
from activations is $
M _"act" ^"Attention" & = B L S  ( (5p+1)D + (2p+1)A S  )  .
$<eq_att_actmem_vanilla>

===== MLP Activations
<mlp-activations>
First we pass the -shaped inputs into the first MLP layer. These turn
into the inputs of the non-linearity, whose same-shaped outputs are then
passed into the last layer, summing to $(2 E + 1) B D S$ total elements
thus far. Adding in the dropout mask, the total memory requirement
across all layers is: $
M _"act" ^MLP & = (2 E p+p+1)B D L S .
$<eq_mlp_actmem_vanilla>

===== LayerNorm, Residual Connections, and Other Contributions
<layernorm-residual-connections-and-other-contributions>
Then the last remaining components. The instances each have $B D S$
inputs and there are two per transformer block, so
$M _"act" ^LN = 2 p B D L S$, and there is an
additional instance at the end of the architecture#footnote[Following
@korthikanti2022reducing we will neglect this in the below sum, an
$cal(O) ( 1/L
)$ error]. There are two residual connections per block, but
their inputs do not require caching (since their derivatives are
independent of inputs). Then, there are additional contributions from
pushing the last layer's outputs through the language-model head and
computing the loss function, but these do not scale with $L$ and are
ultimately $ cal(O)
(  V /( D L ) )$ suppressed, so we neglect them.

===== Total Activation Memory
<total-activation-memory>
Summing up the contributions above, the total activation memory cost
per-layer is $
M _"act" ^"total" & approx 2B D L S  ( p(E+4) + 1 + cal(O) (
V/( D L ) )  )
+ A B L S ^2  ( 2p+1 )  .
$<eq_act_mem_total_no_sharding> Evaluating in common limits, we have:
$
  M_"act"^"total" |_( E=4, p=4 ) & =B L S ( 66 D+15 A S ) \
  M_"act"^"total" |_( E=4, p=2 ) & =B L S ( 34 D+5 A S )
$

==== When does mixed-precision reduce memory?
<when-does-mixed-precision-reduce-memory>
(Answer: usually.) We saw in @sec_params_grads_optim_mem that mixed
precision #emph[increases] the fixed costs of non-activation memory, but
from the above we also see that it also #emph[reduces] the activation
memory and the saving increase with larger batch sizes and sequence
lengths. It is straightforward to find where the tipping point is.
Specializing to the case $E = 4$, vanilla mixed-precision case with no
parallelism#footnote[With both tensor- and sequence-parallelism, the
parallelism degree $T$ actually drops out in the comparison (since both
form of memory are decrease by $1 \/ T$, so this restriction can be
lifted.], the minimum batch size which leads to memory savings is
$
  B_"min" & = ( 6 D^2 ) / ( 8 D S + A S^2 ) .
$<eq_min_mp_batch_size> Plugging in numbers for the typical
$cal(O) ( 40  "GiB" )$ model in the Summer of 2023
gives $B _"min"  cal(O) ( 1)$, so
mixed-precision is indeed an overall savings at such typical scales.

#block[
  Side Note: Optimizations

  The above analysis is conservative and accounts for more tensors than
  are actually saved in practice.

  For instance, both the input and outputs of all non-linearities were
  counted, but there are many activations whose derivatives can be
  reconstructed from its outputs alone: $phi' (z) = F (phi (z))$
  for some $F$. Examples:

  - `ReLU`: since $phi (z) = z theta (z)$, then (defining the derivative at
    zero to be zero) $phi' (z) = theta (z) = theta (phi (z))$.
    Correspondingly, torch only uses the outputs
    #link("https://github.com/pytorch/pytorch/blob/73d288fdf9d0beb76229cabc8566ee116f8a21a2/tools/autograd/derivatives.yaml#L2009-L2011")[to compute the derivative]
    (there is no self arg in the line).

  - $tanh$: since $tanh' (z) = 1 - tanh (z)^2$.

  Other cases do not have this nice property, in which case both the
  inputs and outputs need to be stored:

  - `GeLU` @hendrycks2023gaussian: $phi (z) = z Phi (z)$ here and the
    derivative
    $phi' (z) = Phi (z) + frac(z e^(- z^2 \/ 2), sqrt(2 pi))$, both
    the inputs and outputs
    #link("https://github.com/pytorch/pytorch/blob/73d288fdf9d0beb76229cabc8566ee116f8a21a2/tools/autograd/derivatives.yaml#L2041-L2044")[must be used in the backwards pass.].

    The explicit CUDA kernel
    #link("https://github.com/pytorch/pytorch/blob/73d288fdf9d0beb76229cabc8566ee116f8a21a2/aten/src/ATen/native/cuda/ActivationGeluKernel.cu#L70-L84")[is here].

  If the inputs in each of these cases are not needed for any other part
  of the backwards pass, they are garbage collected in soon after
  creation.

  ===== Example
  <example>
  $SM$ is another instance where this occurs, since $
  partial _( i ) SM  ( x _( j )  ) &= delta _( i j )SM  ( x _( j )  ) - SM  ( x _( i )  ) SM  ( x _( j )  )
  $<eq_softmax_derivative> Because of this, the actual amount of activation
  memory due to the attention layer after the forwards pass is
  @eq_att_actmem_vanilla with
  $2 p arrow.r p$ in the $cal(O) ( S
  ^2 )$ term, though the above expression better reflects the
  necessary peak memory.

]


== Training FLOPs <sec_flops_training>
The total number of floating point operations (FLOPs)#footnote[The
notation surrounding floating-point operations is very confusing because
another quantity of interest is the number of floating-point operations
a given implementation can use #emph[per-second]. Some times, people use
FLOPS or FLOP/s to indicate the rate, rather than the gross-count which
has the lower case “s\", FLOPs, but there's little consistency in
general. We will use FLOPs and FLOP/s.] needed to process a given batch
of data is effectively determined by the number of matrix multiplies
needed.

Recall that a dot-product of the form $v dot.op M$ with $v in bb(R)^m$
and $M in bb(R)^(m \, n)$ requires $(2 m - 1) times n approx 2 m n$
FLOPs . For large language models,
$m,n  cal(O) ( 10 ^3 )$, meaning that even
expensive element-wise operations like acting on the same vector $v$
pale in comparison by FLOPs count #footnote[Since their FLOPs counts
only scales as $ cal(O) ( n )$ where the omitted
constant may be relatively large, but still negligible when all
dimensions are big.]. It is then a straightforward exercise in counting
to estimate the FLOPs for a given architecture. The input tensor is of
shape throughout.

#block[
  Essentials The number of FLOPs to push a batch of $B$ of sequence-length
  $S$ examples through the forwards-pass of a decoder-only transformer is
  approximately $2 B S N _"params"$ where the number of
  parameters accounts for any reductions due to tensor- and
  sequence-parallelism#footnote[A quick argument: a computation of the
form $T_(a_0 dots.h a_n j) = V_(a_0 dots.h a_A i) M_(i j)$ requires
$2 A_0 dots.h A_n I J$ FLOPs where the capital letters represent the
size of their similarly-index dimensions. Thus, the FLOPs essentially
count the size of the matrix $M$ (that is, $I J$), up to a factor of 2
times all of the dimensions in $V$ which weren't summed over. Therefore,
passing a -shaped tensor through the Transformer architecture would give
$tilde.op 2 B S times$(sum of sizes of all weight-matrices) FLOPs, and
that this last factor is also approximately the number of parameters in
the model (since that count is dominated by weights). Thus, FLOPs
$approx 2B S N _(
                "params"  )$. This is the correct as long as the
self-attention FLOPs with
$cal(O) ( S ^2 )$-dependence which we didn't account
for here are actually negligible (true for $S lt.tilde 10 D$).]. The
  backwards-pass costs about twice as much as the forwards-pass. This is
  true as long as $S lt.tilde D$).

]

=== No Recomputation
<no-recomputation>
Start with the case where there is no recomputation activations. These
are the #strong[model FLOPs] of @korthikanti2022reducing, as compared to
the #strong[hardware FLOPs] which account for gradient checkpointing.

==== : Forwards
<forwards>
The FLOPs costs:

- Generating the query, key, and value vectors: $6 B S D^2$

- Attention scores: $2 B D S^2$

- Re-weighting values: $2 B D S^2$

- Final projection: $2 B S D^2$

==== : Forwards
<forwards-1>
Passing a through the layer, the FLOPs due to the first and second
matrix-multiplies are equal, with total matrix-multiply FLOPs
$4 B S E D^2$.

==== Backwards Pass: Approximate
<backwards-pass-approximate>
The usual rule of thumb is to estimate the backwards pass as costing
twice the flops as the forwards pass. This estimate comes from just
counting the number of $cal(O) ( n ^2 )$
matrix-multiply-like operations and seeing that for every one matrix
multiplication that was needed in the forward pass, we have roughly
twice as many similar operations in the backwards pass.

The argument: consider a typical sub-computation in a neural network
which is of the form $z' = phi (W dot.op z)$ where $z' \, a$ are
intermediate representations $z \, z'$, $phi$ is some non-linearity,
and where the matrix multiply inside the activation function dominates
the forwards-pass FLOPs count, as above. Then, in the backwards pass for
this sub-computation, imagine we are handed the upstream derivative
$partial _( z \' ) cal(L)$. In order to complete backpropagation,
we need both to compute $partial _( W )cal(L)$ to update $W$ and
also $partial _( z ) cal(L)$ to continue backpropagation to the
next layer down. Each of these operations will cost about as many FLOPs
as the forwards-pass, hence the estimated factor of two (but, as we will
see, this is a very rough estimate).

Being more precise, let $z$ be -shaped and let $W$ be -shaped such that
it acts on the last index of $z$, making $z'$ -shaped. Denoting
$D = product_i D_i$ be the number of elements along the $D_i$ directions
for brevity, the forward-FLOPs cost of the sub-computation is therefore
$2 D I J$.

Adding indices, the two derivatives we need are $
( partial cal(L) )/( partial W _( i j ) ) & = ( partial cal(L) )/( partial z \'_( d _( 0 ) ... d _( n )i ) )phi\'  ( ( W dot z  ) _( d _( 0 )... d _( n )i )  )
z _( d _( 0 )... d _( n ) j ) \
( partial cal(L) )/(partial z _( d _( 0 )... d _( n )j ) ) & = ( partial cal(L)
)/( partial z \'_( d _( 0 ) ... d _( n )i ) )phi\'  ( ( W dot z  ) _( d _(
0 )... d _( n )i )  ) W _( i j ) ,
$<eq_backprop_derivatives>
which have shapes and , respectively. On the right
side, $z$ and $W dot.op z$ are cached and the element-wise computation
of $phi' (W dot.op z)$ has negligible FLOPs count, as discussed
above: its contribution is $cal(O)
( 1/I )$ suppressed relative to the matrix-multiplies. The
FLOPs count is instead dominated by the broadcast-multiplies, sums, and
matrix-products.

The two derivatives in @eq_backprop_derivatives each have the same first two factors in common, and
it takes $D I$ FLOPs to multiply out these two -shaped tensors into another result with the same
shape. This contribution is again $cal(O) ( 1/I)$ suppressed and hence negligible. Multiplying this
factor with either $z_(d_0 dots.h d_n i)$ or $W_(i j)$ and summing over the appropriate indices
requires $2 D I J$ FLOPs for either operation, bringing the total FLOPs to $4 D I J$, which is
double the FLOPs for this same sub-computation in the forward-direction, hence the rough rule of
thumb#footnote[Note also that the very first layer does not need to perform the second term in
  @eq_backprop_derivatives;, since we do not need to backpropagate to the inputs, so the total
  backwards flops is more precisely $4 D I J (L - 1) + 2 D I J$.].

==== Backwards Pass: More Precise
<backwards-pass-more-precise>
#strong[TODO]

==== Total Model FLOPs
<total-model-flops>
The grand sum is then#footnote[With a large vocabulary, the cost of the
final language model head matrix multiply can also be significant, but
we have omitted its $L$-independent, $2 B D S V$ contribution here.]:
$
  C^"model" & approx 12 B D L S ( S + ( 2+E )D ) .
$<eq_model_flops>
We can also phrase the FLOPs in terms of the number
of parameters
@eq_approx_params_tensor_parallel
as $
C ^"model" | _( T=1 ) & = 6B S N _"params" times  ( 1 + cal(O) ( S/D)  )
$ where we took the $T = 1 \, D gt.double S$ limit for
simplicity and we note that $B S$ is the number of total tokens in the
processed batches.

=== Training Time <sec_train_time>
Training is generally compute bound (see App.~@app_compute_mem_bound)
and based on the results of @sec_flops_training the quickest one
could possibly push a batch of data through the model is
$
  t_"min" & = ( C^"model" ) / ( lambda_"FLOP/s" ) .
$<eq_tmin_model>
Expanding to the entire training run, then with
perfect utilization training will take a time $
t _"total" & approx (6N _"params" N _"tokens")/( lambda _"FLOP/s" ) .
$<eq_training_rule_of_thumb>
Adjust $lambda _"FLOP/s"$ to the actual
achievable FLOP/s in your setup to get a realistic estimate.

How many tokens should a model of size $N _"params"$?
Scaling laws (@sec_scaling_laws) provide the best known answer, and
the Summer 2023 best-guess is that we optimally have
$N _"tokens"approx 20 N _"params"$. So that the
above is $
t _"total" & approx (120N _"params" ^2)/( lambda _"FLOP/s" ) ,
$ leading to quadratic growth in training time.

Note that the above is only correct if we are actually only spending
$C ^"model"$ compute per iteration. This is not correct if
we use gradient checkpointing and recomputation, in which case we
alternatively spend true compute
$C ^"hardware" \> C ^"model"$, a distinction
between #strong[hardware FLOPs] and #strong[model FLOPs]. Two
corresponding efficiency measures are #strong[model FLOPs utilization]
(MFU) and #strong[hardware FLOPs utilization] (HFU). If our iterations
take actual time $t _"iter"$, then these are given by
$
  "MFU" & = ( t_"iter" ) / ( t_"min"^"model" ) , quad "HFU" = ( t_"iter" ) / ( t_"min"^"hardware" ) ,
$<eq_mfu> where $t _"min" ^"model"$ is
@eq_tmin_model and
$t _"min" ^"hardware"$ is similar but using
$C ^"hardware"$.

=== Scaling Laws <sec_scaling_laws>
Empirically-discovered scaling laws have driven the race towards larger
and larger models.

#block[
  Essentials Decoder-only model performance improves predictably as a
  function of the model size, dataset size, and the total amount of
  compute. As of Summer 2023, there is little sign of hitting any kind of
  wall with respect to such scaling improvements.

]
The central parameters are:

- The number of non-embedding model parameters, as excising embedding
  params was found to generate cleaner scaling laws. Because our
  $N _"params"$ has already been typically neglecting these
  parameters, we will just use this symbol in scaling laws and keep the
  above understanding implicit.#footnote[Presumably, the scaling laws
  are cleaner with these neglected because these params do not
  contribute directly to FLOPs, unlike most other parameters.]
  @kaplan2020scaling.

- $C$: total compute, often in units like PFLOP/s-days $tilde.op 10^20$
  FLOPs

- $N _"tokens"$: dataset-size in tokens

- $cal(L)$: cross-entropy loss in nats

The specific form of any given scaling law should also be understood to
apply to a pretty narrowly defined training procedure, in which choices
like the optimizer, learning-rate scheduler, hyperparameter search
budget, vocabulary size, tokenization, etc. are often rigidly set.
Changing different components of the training procedure is liable to
create different scaling laws (though nice laws of some form are still
expected to exist).

==== Original Scaling Laws
<original-scaling-laws>
The first scaling-laws were reported in @kaplan2020scaling. Their
simplest form relates the value of the cross-entropy loss #emph[at
convergence] (and in nats), $cal(L)$, to the number of non-embedding
parameter, dataset size in token, and the amount of compute, #emph[in
the limit] where only one of this factors is bottlenecking the
model#footnote[Unclear to me how you know when this is the case?]. The
laws (in our notation):

- $cal(L) (N _"params") approx  ( N _"params"^( star ) / N _( "params"
  )  ) ^( alpha _( N ) )$, with
  $alpha_N approx 0.076$ and
  $N_"params"^( star ) approx
    8.8 times 10^( 13 )$

- $cal(L) (N _"tokens") approx  ( N _( "tokens)" ^( star ) / N _( "tokens"
  )  ) ^( alpha _( T ) )$, with
  $alpha_T approx 0.095$ and
  $N_("tokens" )^( star ) approx
    5.4 times 10^( 13 )$

- $cal(L) (C) approx  ( C ^( star ) / C
   ) ^( alpha _( C ) )$, with
  $alpha_C approx 0.050$ and $C^star.op approx 3.1 times 10^8$
  PFLOP/s-days, where the batch size was assumed to be chosen to be
  compute optimal per the criteria they outline



#figure(
  image("figures/SimplePowerLaws.png"),
  caption: [
    Original scaling laws from @kaplan2020scaling.
  ],
)
<fig_scaling_laws_original_1>

#figure(
  image("figures/EfficiencyIllustration.png"),
  caption: [
    From @kaplan2020scaling. Larger models are much more
    sample-efficient (faster).
  ],
)
<fig_scaling_laws_original_2>

==== Chinchilla Scaling Laws
<chinchilla-scaling-laws>
As of Summer 2023, the Chinchilla scaling laws in @hoffmann2022training
are the de facto best scaling laws for guiding training. The central
difference between @hoffmann2022training and @kaplan2020scaling is that
in the former they adjust their cosine learning-rate schedule to reflect
the amount of planned training, while in the latter they do
not#footnote[The learning-rate schedule consist of a linear warm-up
stage from a very small $eta$ up to the largest value
$eta _"max"$, after which the cosine bit kicks in:
$eta (s)= eta _"min" +  ( eta _"max" - eta  _"min"  ) times cos  (( pi s )/( 2 s _"max" )  )$
with $s$ the step number. In Fig.~A1 of @hoffmann2022training they
demonstrate that having the planned $s
            _"max"$ duration of the scheduler be longer than
the actual number of training steps is detrimental to training (they do
not study the opposite regime), which is effectively what was done in
@kaplan2020scaling. Probably the more important general point is again
that the precise form of these scaling laws depend on details of fairly
arbitrary training procedure choices, such as the choice of
learning-rate scheduler.].

Several different analyses are performed which all give very similar
results. The outputs are the optimal values of
$N _"params", N _"tokens"$ given a compute budget
$C$.

- They fix various buckets of model sizes and train for varying lengths.
  In their resulting loss-vs-FLOPs plot, they determine the model size
  which led to the best loss at each given FLOPs value, thereby
  generating and optimal model size vs compute relation.

- They fix various buckets of FLOPs budget and train models of different
  sizes with that budget, finding the optimal model size in each case. A
  line can then be fit to the optimal settings across FLOPs budgets in
  both the parameter-compute and tokens-compute planes.

- They perform a parametric fit to the loss#footnote[In
  @hoffmann2022training they model the scaling of the test loss, while
  in @kaplan2020scaling they use the training loss.]:
  $
    cal(L) (N_"params", N_"tokens") & =E + A / ( N_"params"^( alpha ) ) + B / ( N_"tokens"^( beta ) ) ,
  $<eq_chinchilla> fit over a large range of parameter and token
  choices. The best-fit values are:
  $
    E & = 1.69 med \, quad A = 406.4 med \, quad B = 410.7 med \, quad alpha = 0.34 med \, quad beta = 0.28 med .
  $
  Using $C approx 6 N _( "params)" N _"tokens"$, the
  above can be minimized at fixed compute either for number of parameter
  or the size of the dataset.

In all cases, the findings are that at optimality $N _"params"  N _( "tokens") C ^( .5 )$: both
the parameter and tokens budget should be scaled in equal measure.

= Fine Tuning
<fine-tuning>
== Instruction Fine Tuning
<instruction-fine-tuning>
Generally, instruction fine-tuning is a follow-on step after model
pre-training#footnote[A terminology note: pre-training is standard
next-token training on an enormous, general dataset, supervised
fine-tuning typically indicates additional, subsequent training on a
higher-quality, maybe domain-specific dataset, and instruction
fine-tuning follows.]. The pre-training, pure next-token prediction
task is altered to optimize an objective which now incorporates other
data, typically information regarding human preferences#footnote[One
failure mode this corrects for: next-token training would do best by
replicating common mistakes in grammar or statements of fact which can
be corrected for using these methods.].

=== Direct Preference Optimization <subsec_dpo>
Direct Preference Optimization (DPO)
@rafailov2024directpreferenceoptimizationlanguage is a vast
simplification of previous reinforcement-learning based methods (namely
PPO-based ones @schulman2017proximalpolicyoptimizationalgorithms).

DPO aims to solve the RLHF optimization problem defined over a dataset
$cal(D)  (x, y _( l ), y
_( w ))$ corresponding to prefixes ($x$) and pairs of preferred and
dispreferred completions#footnote[I guess the $l \, w$ subscripts are
for \"lose\" and \"win\"?] ($y_l \, y_w$). The relevant components are:

+ A baseline language model: $pi _"ref" (y|x)$, usually a supervised fine-tuned model trained on
high-quality data.

+ The to-be-trained model: $pi_theta (y \| x)$, usually initialized to $pi _"ref" (y|x)$. This is
the #emph[policy] in the literature.

+ A reward model which produces $p (y_w succ y_l \| x)$, the probability#footnote[Whether one
    completion is preferred over another is a probabalistic question since, e.g., not everyone in
    the population will agree.] $y_w$ is favored over $y_l$. The reward function $r (x \, y)$
reflects how well $y$ completes the prefix $x$, in this context, and we assume the probability can
be expressed in terms of the reward function $p (y_w succ y_l \| x) = p (r (x \, y_w) \, r (x \,
y_l))$. The reward model is commonly an LLM with a scalar output head attached.

First, a quick review of RLHF, which proceeds in stages. First,
$cal(D)$ is used to train a reward model informed by the dataset
$cal(D)$. The optimal reward model $r_star.op$ minimizes the binary
cross-entropy loss over $cal(D)$, which is just
$
  cal(L)_( r ) &= -E_( x, y_( l ), y_( w ) cal(D) ) ln p(y_( w ) succ y_( l )| x ) .
$<eq_rlhf_reward_loss>
The reward model embodies human preferences and we
want to transfer this knowledge to the language model $pi_theta$. This
can be done by optimizing $pi_theta$ to generate completions of inputs
that lead to large rewards, reflecting human-preferred generations. In
order to also keep the model from straying too far from its reference
base, a tunable KL-divergence penalty is also added#footnote[We've
written the above as a loss so that we're minimizing everywhere.]:
$
  cal(L)_"RLHF" &= E_(x cal(D), y pi_( theta )(y|x) ) ( -r_( star )
  (x, y) + beta D_"KL" ( pi_( theta )(y|x)|| pi_("ref))(y|x" ) ) .
$<eq_rlhf_loss>
Reinforcement-learning methods are typically used to
optimize the $pi_theta$ model and the generation step is particularly
costly. In particular, the usual gradient-based optimization methods
cannot be used because the loss depends on generated tokens which are
discontinuous (non-differentiable) functions of the model's parameters.

DPO improves upon RLHF by skipping any generation step, removing the explicit reward function, and
making the optimization problem amenable to gradient based methods by choosing a specific functional
relation between the reward function $r (x \, y)$ and the preference probability $p (y_w succ y_l \|
x)$. Whereas RLHF minimizes the loss $cal(L)_"rlhf"$ @eq_rlhf_loss
subject to a fixed, optimal reward function found by first minimizing the reward loss $cal(L) _( r
)$ @eq_rlhf_reward_loss;, DPO is essentially derived in the
opposite direction: first, find the functional form of $pi_theta$ which minimizes the RLHF loss for
an arbitrary reward function, and then use this form when minimizing of the cross-entropy defining
the reward function#footnote[This is analogous to minimizing the regular function $f (x \, y)$
subject to also minimizing $g (x)$. This can either be done by solving the second for
$x_star.op$ and minimizing $f (x_star.op \, y)$ (the RLHF strategy), or first solving
$frac(partial f, partial y) = 0$ to find $x_star.op (y)$ and then minimizing $g (x_star.op (y))$
(the DPO strategy).].

The $pi_theta$ which minimizes the RLHF loss
@eq_rlhf_loss for an arbitrary reward
function $r (x \, y)$ is given by#footnote[This is easy to show using
the calculus of variations, though it's not the route taken in the
paper. The explicit RLHF loss is
$cal(L) _"RLHF" = integral dif x thin dif y thin p(x) pi _( theta  )(y|x) ( -r(x,y) +beta ln pi _(
theta )(y|x) /pi _"ref"(y|x)  )$ and we want to
minimize this subject to the constraint that $pi_theta (y \| x)$ is
properly normalized. So, we use a Lagrange multiplier and extremize
$cal(L)\'  = cal(L) _"RLHF"+ integral dif x thin dif y thin lambda (x) pi _( theta
)(y|x)$. Solving
$( delta cal(L) \' )/( delta pi _( theta  ) (y|x)) =0$
yields @eq_dpo_soln;.]
$
  pi_(theta)(y|x) &= ( pi_"ref" (y|x)e^( r(x, y) / beta ) ) / ( Z(x) ) ,
$<eq_dpo_soln>
where
$Z(x) = integral dif y thin pi_"ref"(y|x)e^( r(x, y) / beta )$
is a intractable normalization (partition function) factor. However, if
$p (y_w succ y_l \| x)$ only depends on $r (x \, y_w)$ and
$r (x \, y_l)$ through their difference#footnote[In
@rafailov2024directpreferenceoptimizationlanguage, the DPO symmetry
$r (x \, y) arrow.r r (x \, y) + f (x)$, for arbitrary $f (x)$, is said
to induce an equivalence class relation between different reward
functions.], these factors cancel out. Letting
$p (y_w succ y_l \| x) = sigma (r (x \, y_w) - r (x \, y_l))$, for
some#footnote[In the specific case where $sigma$ is the sigmoid
function, this is known as the Bradley-Terry model.] $sigma$, and
eliminating the reward function in the cross-entropy loss via
@eq_dpo_soln reduces $cal(L) _( r )$ to
$
  cal(L)_"DPO" &= -E_( x, y_( l ), y_( w ) cal(D) ) ln sigma (beta (ln ( pi_(
      theta )(y_( w )|x ) ) / ( pi_"ref"(y_( w )|x ) )-ln ( pi_( theta )(y_( l )|x) ) / ( pi_"ref"(y_( l)|x) ) ) ) ,
$<eq_dpo_reward_loss>
which we've now renamed the DPO loss. The loss
@eq_dpo_reward_loss can now be
minimized by standard, gradient based methods without any generation
step.

=== KTO: Preference Finetuning without Pairs <subsec_kto>
DPO requires a dataset of triplets: a prefix, one preferred completion,
and one dispreferred completion. KTO alignment
@ethayarajh2024ktomodelalignmentprospect attempts to reduce the inputs a
prefix, a completion, and a binary signal indicating whether the output
is desirable or not, since such datasets are easier to construct.

The method is based on the ideas of Kahneman and Tversky and the central
ingredient is a value function which monotonically maps outcomes to
perceived values $v : cal(Z) arrow.r bb(R)$, with $cal(Z)$ the space of
outcomes. Some normalization point $z_0$ defines the boundary between
positive and negative outcomes, the value function#footnote[Which can be
taken to satisfy $v (0) = 0$.] is taken to be a function of $z - z_0$,
and human value functions are known to be convex for $z > z_0$
(diminishing returns) and exhibit loss aversion#footnote[Which I suppose
means that $v (z - z_0) + v (z_0 - z) lt.eq 0$ for $z > 0$.].

KTO applies this framework to the usual text-prediction problem as in
the following. The space of outcomes $cal(Z)$ is the reward function
value taken to be $
r _( theta )(x, y)&eq.triple ln ( pi _( theta )(y|x) )/( pi _"ref")(y|x )  ,
$<eq_kto_reward>
the difference in reference and model surprisal, as
inspired by DPO. The reference point is just the expected value of the
reward function over prefixes and trainable-model-generated completions,
i.e., the KL divergence averaged over prefixes:
$
  z_( 0 ) & eq.triple E_( y pi_( theta )(y|x) , x D )r_( theta )(x, y) =E_( x D ) D_"KL"(pi_( theta )(y|x)|| pi_"ref"(y|x)) .
$<eq_kto_ref_pt>
Splitting the space of completions into desirable and
undesirable ones, $cal(Y) = cal(Y)_D union cal(Y)_U$, the KTO
loss#footnote[They also add a constant term to the loss for
normalization purposes which we have omitted. The KTO loss falls into
the broader category of Human Aware Loss Objectives (HALOs) which are a
general class of objectives that roughly fit into the Kahneman-Tversky
form. See the paper for a further discussion and comparison of HALO vs
non-HALO methods.] is taken to be: $
cal(L)_"KTO" &= - E _( x, y  D )v(r _( theta )(x, y) - z _( 0 ))\
v(r _( theta )(x, y)-z _( 0 ))&eq.triple cases(
lambda _( D )sigma  ( beta  ( r _( theta )(x, y) - z _( 0 )  )  ) & y in cal(Y)_( D )\
lambda _( U )sigma  ( beta  ( -r _( theta )(x, y) + z _( 0 )  )  ) & y in cal(Y)_( U )
)
$<eq_kto_loss>
for hyperparameters#footnote[Risk aversion would seem
to require $lambda_U > lambda_D$, but the KTO paper empirically finds
that the opposite regime performs better.]
$beta \, lambda_D \, lambda_U in bb(R)^(+)$ and where $sigma$ is the
sigmoid function. So, $v (r_theta (x \, y) - z_0)$ is maximized by
sending $r_theta arrow.r oo$ for desirable results and to $- oo$ for
undesirable ones, while the normalization point $z_0$ concentrates
updates on examples whose rewards do not stray wildly from the average
reward, which implicitly carries information about the reference model.

The reference point $z_0$ @eq_kto_ref_pt
is a problem, because it requires generation which is both expensive and
not differentiable (the problem DPO solves). So, the authors perform a
rough estimate of the scale and do not backpropagate through $z_0$,
(which is a bit questionable).

= Parallelism
<parallelism>
The simplicity of the Transformers architecture lends itself to a deep
variety of parallelism strategies. We review some of them below.

== Tensor Parallelism <subsec_tensor_parallelism>
#block[
  Side Note: I wrote a blog post on this
  #link("https://www.determined.ai/blog/tp")[here.]

]
In #strong[Tensor Parallelism], some times also called #strong[Model
Parallelism], individual weight matrices are split across devices
@shoeybi2020megatronlm. We consider the and layers in turn. Assume
$T$-way parallelism such that we split some hidden dimension into
$T$-equal parts across $T$ workers#footnote[All $T$ workers work on
processing the same batch collectively. With $N > T$ workers, with $N$
perfectly divisible by $T$, there are $N \/ T$ different data parallel
groups. Critical-path TP communications occur within each data parallel
group and gradients are synced across groups. Ideally, all the workers
in a group reside on the same node, hence the usual $T = 8$.]

#block[
  Essentials The cost of large weights can be amortized by first sharding
  its output dimension, resulting in differing activations across group
  members. Later, the activations are brought back in sync via a . Weights
  which act on the sharded-activations can also be sharded in their input
  dimension. In the backwards pass, another is required.

]

=== MLP
<mlp-1>
It is straightforward to find the reasonable ways in which the weights
can be partitioned. We suppress all indices apart from those of the
hidden dimension for clarity.

The first matrix multiply $z_d W_(d e)^0$ is naturally partitioned
across the output index, which spans the expanded hidden dimension
$e in (0 \, dots.h \, E D - 1)$. This functions by splitting the weight
matrix across its output indices across $T$ devices:
$W_(d e)^0 = W_(d (f t))^0 equiv macron(W)_(d f macron(t))^0$ (again in
\-like notation, with bars denoting that the tensor and particular
indices are sharded; see App.~@app_conventions), where in the split
weights $macron(t) in (0 \, dots.h \, T - 1)$, and
$f in (0 \, dots.h \, frac(E D, T) - 1)$. Each of the $T$ workers
compute one shard of $z_d macron(W)_(d f macron(t))^0$, i.e. each has a
different value of $macron(t)$.

Let the partial outputs from the previous step be $macron(z)_(f t)$
(batch-index suppressed), which are -shaped, with the final dimension
sharded across workers. The non-linearity $phi$ acts element wise,
and using the updated $macron(z)_(f macron(t))$ to compute the second
matrix multiply requires a splitting the weights as in
$W_(e d')^1 = W_((f t) d')^1 equiv macron(W)_(f macron(t) d')^1$
(dividing up the incoming $e$ dimension), such that the desired output
is computed as in
$macron(z)_(f macron(t)) dot.op macron(W)_(f macron(t) d')^1$, sum over
$macron(t)$ implied. Each device has only $macron(t)$ component in the
sum (a -shaped tensor) and an is used to give all workers the final
result. This is the only forward-pass collective
communication#footnote[The amount of communicated data is
$cal(O) ( B S D )$.].

One-line summary of the parallel decomposition:
$
  z_(s d') arrow.l phi (z_d W_(d e)^0) W_(e d')^1 & = phi (z_d macron(W)_(d f macron(t))^0) macron(W)_(f macron(t) d')^1 med .
$
The progression of tensor shapes held by any single worker is
+ `(B, S, D)`
+ `(B, S, E*D/T)`
+ `(B, S, D)`

In the backwards pass, another (see App.~@app_collective_communications)
is needed for proper gradient computations with respect to the first
layer's outputs. This is true whenever an operation producing a sharded
output involved non-sharded tensors: if an operation
$macron(y)_(macron(r)) = F (x \, dots.h)$ produces a sharded output from
an unsharded in put $x$ (all other indices suppressed), the derivative
with respect to $x$ requires a sum over ranks,
$( partial cal(L) )/( partial x ) = ( partial cal(L) )/( partial macron(y) _(
macron(r) ) ) ( partial macron(y) _( macron(y) ) )/( partial x )$.
Note that each worker will have to store all components of the input $z$
for the backward pass.

#figure(
  image("figures/mlp_mp_2.png"),
  caption: [
    Tensor parallelism for the layers. Graphic from
    @shoeybi2020megatronlm. The $f \/ g$ operations are the collective
    identity/ operations in the forwards pass and the /identity
    operations in the backwards pass.
  ],
)
<fig_mlp_tensor_parallel>

=== Attention
<attention>
Because the individual attention head computations are independent, they
can be partitioned across $T$ workers without collectively
communications. An is needed for the final projection, however, which
results in the various re-weighted values $y_(b s e a)$
@eq_reweighted_values;.

To review, the attention outputs $z'_(s d)$ generated from inputs
$z_(s d)$ can be expressed as $
z\' _( s e a ) &= MHA(q _( s e a ), k _( s e a ), v _( s e a )) O _( e a d )\
$ where:

- We have split the $d$-index as in $z_(s d) arrow.r z_(s (e a))$ with
  $e$ and $a$ the head-dimension and head-index

- $q_(s e a) \, k_(s e a) \, v_(s e a)$ are the query, keys and values
  derived from the inputs

- $MHA$ is the multi-head attention function, whose outputs are
  the same shape as its value inputs

- The dual sum over head-dimension index ($e$) and attention-head-index
  ($a$) is the sum-and-concatenate step from the more explicit
  description in @attn_layer

- and biases were ignored for simplicity

In order to parallelize the above $T$-ways, we simply shard across the
dimension $a$ which indexes the different attention heads. The
$MHA$ computations all process in embarassingly-parallel
fashion, and an all-reduce is needed to complete the sum over the
$a$-index across devices.

The collective communications story is essentially equivalent to that of the layers#footnote[The
amount of communicated data is again $cal(O) ( B S D)$.]: one is needed in the forwards pass and
one in the backwards-pass.

The progression of tensor shapes held by any single worker is
+ `(B, S, D)`
+ `(B, S, D/A, A/T)`
+ `(B, S, D)`

It is worth comparing the communications and FLOPs costs of these
sharded layers. Each layer costs
$cal(O) ( B S ( 4+2E ) D^2 / T )$
FLOPs and communicates $cal(O) ( B S D )$ bytes and so the
communication-to-compute-time ratio is $
( t _"compute" )/( t _"comms" ) &  (  ( 4+2E ) D )/( T ) times ( lambda _"comms" )/( lambda _"FLOP/s")  .
$ Since#footnote[Assuming
$lambda _"FLOP/s" $100 TFLOP/s and
$lambda _"comms"
$ 100 GiB/s.]
$( lambda _"comms" )/( lambda _"FLOP/s")  10 ^( -3 )$FLOPs/B,
communication and compute take similar times when
$D  cal(O) ( 10 ^3 )$ for typical setups with
$T  cal(O) ( 10 )$ and so tensor-parallelism
requires $D gt.tilde 10^4$ to reach similar efficiency to the
non-tensor-parallel implementations.

#figure(
  image("figures/attention_mp_2.png"),
  caption: [
    Tensor parallelism for the layers. Graphic from
    @shoeybi2020megatronlm. The $f \/ g$ operators play the same role as
    in Fig.~@fig_mlp_tensor_parallel.
  ],
)
<fig_attn_tensor_parallel>

=== Embedding and LM Head
<embedding-and-lm-head>
Last, we can apply tensor parallelism to the language model head, which
will also necessitate sharding the embedding layer, if the two share
weights, as typical.

For the LM head, we shard the output dimension as should be now
familiar, ending up with $T$ different -shaped tensors, one per group
member. Rather than communicating these large tensors around and then
computing the cross-entropy loss, it is more efficient to have each
worker compute their own loss where possible and then communicate the
scalar losses around#footnote[In more detail, given the gold-answers
$y_(b s)$ for the next-token-targets, a given worker can compute their
contribution to the loss whenever their -shaped output $z_(b s v')$
contains the vocabulary dimension $v_(\*)$ specified by $y_(b s)$,
otherwise those tensor components are ignored.].

For a weight-tied embedding layer, the former construction requires in
order for every worker to get the full continuous representation of the
input.

=== LayerNorm and Dropout
<layernorm-and-dropout>
instances are not sharded in pure tensor parallelism both because there
is less gain in sharding them parameter-wise, but also sharding in
particular would require additional cross-worker communication, which we
wish to reduce as much as possible. layers are also not sharded in where
possible in pure tensor parallelism, but sharding the post-attention
layer is unavoidable. It is the goal of sequence parallelism is to shard
these layers efficiently; see @subsec_seq_parallelism.

=== Effects on Memory
<effects-on-memory>
The per-worker memory savings come from the sharding of the weights and
the reduced activation memory from sharded intermediate representations.

The gradient and optimizer state memory cost is proportional to the
number of parameters local to each worker (later we will also consider
sharding these components to reduce redundantly-held information). The
number of parameters per worker is reduced to $
N _"params" & approx (4 + 2E) ( L D ^2 )/T ,
$<eq_approx_params_tensor_parallel>
counting only the dominant contribution from weights
which scale with $L$, since every weight is sharded. Since all
non-activation contributions to training memory scale with
$N _"params"$, this is a pure $1 \/ T$ improvement.

The per-layer activation memory costs @eq_att_actmem_vanilla and @eq_mlp_actmem_vanilla are altered
to:
$
  M_"act"^"Attention" & = B S ( (p + ( 4p ) / T+1 )D +
    (( 2p+1 ) / T )A S ) \
  M_"act"^MLP & = (( 2E p ) / T +p+1 )B D S .
$<eq_act_mem_attn_mlp>
The derivation is similar to before. Adding in the
(unchanged) contributions from instances, the total, leading order
activation memory sums to $
M _"act" ^"total" & approx 2B D L S  ( p  (2+ ( E+2 )/T  ) + 1  )
+ A B L S ^2  ( ( 2p+1 )/ T  )  .
$<eq_act_mem_total_tensor_parallel>
Again, the terms which did not receive the $1 \/ T$
enhancement correspond to activations from unsharded and instances and
the $1 \/ T$'s improvements can be enacted by layering sequence
parallelism on top (@subsec_seq_parallelism).

== Sequence Parallelism <subsec_seq_parallelism>
In @eq_act_mem_total_tensor_parallel;, not every factor is reduced by $T$. #strong[Sequence
  Parallelism] fixes that by noting that the remaining contributions, which essentially come from
and #footnote[Recall, though, from @layer_norm that the parameters in are completely redundant and
  can simply be removed without having any effect on the expressive capabilities of the
  architecture.], can be parallelized in the sequence dimension (as can the residual connections).

The collective communications change a bit. If we shard the tensors
across the sequence dimension before the first , then we want the
following:

+ The sequence dimension must be restored for the layer

+ The sequence should be re-split along the sequence dimension for the
  next instance

+ The sequence dimension should be restored for the layer #footnote[This
  doesn't seem like a hard-requirement, but it's what is done in
  @korthikanti2022reducing.]

The easiest way to achieve the above is the following.

+ If the tensor parallelization degree is $T$, we also use sequence
  parallelization degree $T$.

+ The outputs of the first are -ed to form the full-dimension inputs to
  the layer

+ The tensor-parallel layer functions much like in
  Fig.~@fig_attn_tensor_parallel #emph[except] that we do not re-form
  the outputs to full-dimensionality. Instead, before the layer, we them
  from being hidden-sharded to sequence-sharded and pass them through
  the subsequent / combination, similar to the first step

+ The now-sequence-sharded tensors are reformed with another to be the
  full-dimensionality inputs to the layer whose final outputs are
  similarly -ed to be sequence-sharded and are recombined with the
  residual stream

The above allows the mask and weights to be sharded $T$-ways, but if we save the full inputs to the
and layers for the backwards pass, their contributions to the activation memory are not reduced (the
$p$-dependent terms in @eq_act_mem_attn_mlp;). In @korthikanti2022reducing, they solve this by only
saving a $1 \/ T$ shard of these inputs on each device during the forward pass and then performing
an extra when needed during the backwards pass. Schematics can be s e n in
Fig.~@fig_tensor_seq_parallel and Fig.~@fig_tensor_seq_parallel_detail below. The activation memory
is then reduced to:
$
  M_"act"^"total" & =( 2B D L S ( p(E+4) + 1 ) / T
  + ( A B L S^2 ( 2p+1 ) ) / T + cal(O) ( B S V ) .
$<eq_act_mem_total_seq_parallel>

In more detail:

- The norms are just linear operations on the $z_(s d)$,
  $z\' _( s d ) =
  "NORM" ( z _( s d )  )$, and so we split and shard
  cross the sequence dimension
  $z_(s d) arrow.r z_((t r) d) equiv macron(z)_(macron(t) r d)$ with the
  TP-index $t$ sharded across devices.

- The residual stream is also sharded across the sequence dimension.

- The sharded outputs $macron(z)_(macron(t) r d)$ must be re-formed to
  create the attention and MLP inputs via an . There is an optimization
  choice here: either the re-formed tensors can be saved for the
  backward pass (negating the $1 \/ T$ memory savings) or they can be
  re-formed via an , at the cost of extra communication.

- Both the MLP and attention layers need to produce final sums of the
  form $macron(y)_(s macron(y) e) macron(O)_(macron(t) e d)$ for some
  intermediate $macron(y)$ and weight $macron(O)$ sharded across the
  TP-dimension $macron(t)$. The outputs are added to the
  sequence-sharded residual stream, and so sum is optimally computed
  through an with final shape
  $macron(z)_(macron(t)' r d) = z_((t ' r) d) = z_(s d) = macron(y)_(s macron(t) e) macron(O)_(macron(t) e d)$.
  This (along with the mentioned above) replace the s from the
  tensor-parallel case and have the same overall communication cost.

#figure(
  image("figures/transformer-tensor-sequence-parallel.jpg"),
  caption: [
    Interleaved sequence and tensor parallel sections. $g$ and
    $macron(g)$ are and in the forward pass, respectively, and swap
    roles in the backwards pass. Graphic from @shoeybi2020megatronlm.
  ],
)
<fig_tensor_seq_parallel>

#figure(
  image("figures/mlp-tensor-sequence-parallel.jpg"),
  caption: [
    Detail of the sequence-tensor parallel transition for the . Graphic
    from @shoeybi2020megatronlm.
  ],
)
<fig_tensor_seq_parallel_detail>

== Ring Attention <subsec_ring_attention>
Ring Attention @liu2023ringattentionblockwisetransformers is roughly a
distributed version of Flash Attention @subsec_flash_attention: it
enables extremely long sequence-length processing by never realizing the
entire $cal(O) ( S ^2 )$ attention scores at once.

It works by sharding over the sequence dimension. Let $z_(s d)$ is the
(batch-dim suppressed) residual stream of a non-sharded
Transformer#footnote[Like in Sec. @subsec_flash_attention, we omit any
normalization factor inside the $SM$.]: $
z _( s d ) &= SM _( s\' )  ( q _( s d\' ) k _( s\'d\' )  ) v _( s\'d )  ,
$ suppressing the causal mask for simplicity of
presentation.

Then in Ring Attention, we shard over $R$ devices via
$z_(s d) arrow.r z_(macron(r) t d)$, and similar for other tensors, to
compute the sharded outputs $
z _( macron(r)t d ) &= SM _( macron(w) x )  ( q _( macron(r)t d\' ) k _( macron(w)x d\' )  ) v _( macron(w)x d )\
&= ( exp  ( q _( macron(r)t d\' ) k _( macron(w)x d\' )  ) )/( sum _( macron(w)\'x\' ) exp  (q _( macron(r)t d\'\' ) k _( macron(w)\'x\' d\'\' ) ) ) v _( macron(w)x d )\
&eq.triple ( Z _(macron(r)t d ) )/( sum _( macron(w)\'x\' ) exp  (q _( macron(r)t d\'\' ) k _( macron(w)\'x\' d\'\' )  ) )\
&eq.triple ( Z _(macron(r)t d ) )/(L _( macron(r)t ))
$ where we introduced some notation which will be
useful blow. Ring Attention is essentially an algorithm for computing
the sharded sums over barred indices via communication. Since the MLP
layers act on every sequence position identically, only the Attention
layers (and the loss computation) require special handling.

The algorithm performs the $macron(w)$ sum as a loop. We present the
simplified case without a causal mask or maximum attention score
tracking. These are important omissions#footnote[See
@brandon2023stripedattentionfasterring for causal mask efficiency
considerations.].

#block[
  Ring Attention (Naive - Missing causal mask/max tracking.) Initialize
  $Z_(macron(r) t d)$, $L_(macron(r) t)$ to zeros Populate the key, query,
  and value shards
  $q_(macron(r) t d) \, k_(macron(w) x d') \, v_(macron(w) x d')$ with
  $macron(r) = macron(w) = r$ on rank $r$ Computing components
  $z_(macron(r) t d)$ $forall t \, d$ prefetch shards
  $k_((macron(w) + 1) x d) \, v_((macron(w) + 1) x d)$ $forall x \, d$
  $Z_(macron(r) t d) arrow.l Z_(macron(r) t d) + exp (q_(macron(r) t d') k_(macron(w) x d')) v_(macron(w) x d)$
  Can use flash attention kernels here
  $L_(macron(r) t) arrow.l L_(macron(r) t) + sum_x exp (q_(macron(r) t d') k_(macron(w) x d'))$
  Can use flash attention kernels here
  $z_(macron(r) t d) arrow.l Z_(macron(r) t d) / L_(macron(r) t)$
  <algo_ring_attn_fwd_naive>

]
At every step in the loop in the algorithm we are computing the sums
$exp (q_(macron(r) t d') k_(macron(w) x d')) v_(macron(w) x d)$ and
$sum_x exp (q_(macron(r) t d') k_(macron(w) x d'))$ for fixed values of
$macron(r) \, macron(w)$ and all values of the other indices. These are
precisely the ingredients that go into the usual attention computation
and for this reason it's possible to use flash attention kernels for
every individual step. implementations of Ring Attention which leverage
flash attention kernels can be found
#link("https://github.com/lucidrains/ring-attention-pytorch")[here] and
#link("https://github.com/zhuzilin/ring-flash-attention")[here].

The full forms of the forwards and backwards passes are again similar to
those of flash attention; see Sec. @subsubsec_fa_details.

==== The Causal Mask
<the-causal-mask>
A naive, row-major sharding of the queries, keys, and vectors is highly
suboptimal for causal attention because it leads to idling GPUs.
Sharding the queries and keys as in $q_s = q_((macron(r) t))$ and
$k_(s') = k_((t ' macron(r) '))$ in row-major order#footnote[That is,
$s = macron(r) T + t$ for $macron(r) in (0 \, dots.h \, R - 1)$ and
$t in (0 \, dots.h T - 1)$ with $S = R T$.], causality means that the
entire chunked attention computation will be trivial for any iteration
in which $r' > r$. This is the case for $R - 1$ iterations for the
$r = 0$ GPU, for instance.

So, we shouldn't shard this way for ring attention. In
@brandon2023stripedattentionfasterring they demonstrate the speed-up
achieved by just reversing the sharding pattern to column-major:
$q_s = q_((t macron(r)))$ and $k_(s') = k_((t ' macron(r) '))$ which
guarantees non-trivial work for every GPU on every iteration, which they
call striped ring attention. In the `ring-flash-attention` repo, they
come up with yet another sharding strategy (\"zig-zag\" attention;
#link("https://github.com/zhuzilin/ring-flash-attention/issues/2")[see this github issue])
which increases efficiency even more. Their strategy can't be naturally
writtein in `einops` notation, but it is easy enough to explain: they
split the sequence length into $2 R$ sequential chunks and give
zero-indexed chunks $r$ and $2 R - r - 1$ to GPU $r$, which ends up
optimally distributing the work.

We analyze the efficiency of each strategy now. Let $q_s$ be sharded to
$q_(macron(r) t)$ according to the strategy's specifics and similar for
$k_s$. On the first iteration every rank has keys and queries with
identical sequence positions, meaning $frac(S / R (S / R + 1), 2)$
positions will be attended to on every rank. The difference comes about
in the subsequent iterations:

+ For naive ring attention, it's all or nothing. $q_(macron(r) t)$ can
  attend to all of $k_(macron(w) x)$ if $macron(r) gt.eq macron(w)$.
  This means that at least one rank needs to perform $S^2 \/ R^2$
  operations every iteration (after the first one), bottlenecking
  processes which have no work.

+ For striped attention ring attention, rank $r = R - 1$ will have
  queries corresponding to positions
  $(R - 1 \, 2 R - 1 \, dots.h \, S - 1)$ and it will always be able to
  attend to $frac(S / R (S / R + 1), 2)$ positions at every iteration,
  just like on the first iteration. This rank (and others which perform
  the same number of operations for some iterations) is the bottleneck,
  since rank $r = 0$, which owns sequence positions
  $(0 \, R \, dots.h \, S - R - 1)$ is only ever able to attend to
  $frac(S / R (S / R - 1), 2)$ positions, since its zero-position
  component can never attend to any thing after the first iteration. This
  mismatch between ranks is suboptimal.

+ For zig-zag attention, there are two scenarios. Each produces the same
  amount of work, which makes this strategy optimal. When
  $macron(r) < macron(w)$ the two sets of position indices covered by
  $t$ on $q_(macron(r) t)$ fall between those#footnote[Example: four
  ranks with sequence length $S = 8$, the rank-zero queries cover
  positions $(0 \, 7)$ while the rank-one keys cover $(2 \, 6)$, so the
  former sandwich the latter.] covered by the $x$ index on
  $k_(macron(w) x)$. That is, every index in the query can attend to
  exactly half of the indices on the keys: $frac(S^2, 2 R^2)$
  operations. In the opposite#footnote[The $macron(w) = macron(r)$ case
  was already treated in the first iteration.] $macron(w) < macron(r)$
  case, the query positions sandwich those of the keys and so the upper
  half of the query positions can attend to all of the key's positions,
  again $frac(S^2, 2 R^2)$ operations. So work is perfectly distributed
  and no rank serves as a bottleneck.

== Pipeline Parallelism <subsec_pipe_parallelism>
TODO

= Vision
<vision>
Notes on the usage of Transformers for vision tasks.

== Vision Transformers <sec_vit>
The original application of the Transformers architecture
@dosovitskiy2021imageworth16x16words divides 2D images into patches of
size $P times P$, e.g. flattening a three-channel $i_(x y c)$ image to
shape $f_(s d)$ where $d in (0 \, dots.h \, P^2 C - 1)$ and the
effective sequence length runs over $s in (0 \, L^2 C \/ P^2 - 1)$, for
an $L times L$ sized image#footnote[Example: for a $256 times 256$,
three-channel image with a $16 times 16$ patch size, the effective
sequence length is 768.]. A linear projection converts the effective
hidden dimension here to match match the model's hidden dimension. These
are known as #strong[Patch Embeddings].

Since there is no notion of causality, no causal mask is needed. A
special token is prepended and used to generate the final
representations $z_(b d)$ for a batch of images. This can be used for
classification, for instance, by adding a classification head. The
original training objective was just that: standard classification
tasks.

== CLIP <sec_clip>
CLIP (Contrastive Language-Image Pre-Training)
@radford2021learningtransferablevisualmodels is a technique for
generating semantically meaningful representations of images. The method
is not necessarily Transformers specific, but the typical
implementations are based on this architecture.

The core of CLIP is its training objective. The dataset consists of
image-caption pairs (which are relatively easy to extract; a core
motivation), the CLIP processes many such pairs and then tries to
predict which images match to which captions. This is thought to inject
more semantic meaning into the image embeddings as compared with, say,
those generated from the standard classification task.

A typical implementation will use separate models for encoding the text
and image inputs. The two outputs are $t_(b d)$ and $i_(b d)$
shaped#footnote[There may also be another linear projection from the
actual model outputs to a common space, too. Obviously, this is also
necessary if the hidden dimensions of the two models differ.],
respectively, with batch and hidden dimensions, and are canonically
trained so that the similarity score between any two elements is a
function of their dot-product.

The original CLIP recipe:

+ Process the text bracketed with and insertions, use a normal
  Transformer architecture#footnote[The original CLIP paper keeps the
  causal mask.], and extract the last output from the token as the text
  embedding:
  $i_(b d) = z_(b s d) #scale(x: 120%, y: 120%)[\|]_(s = - 1)$.

+ Process the image with a vision transformer network.

+ Project to a common dimensionality space, if needed.

+ Compute the logits through cosine similarity:
  $ell_(b b') = i_(b d) t_(b' d) \/ lr(|i_b|) lr(|t_b|)$. These are used
  to define both possible conditional probabilities#footnote[They differ
  by what is summed over the in the denominator, i.e., which dimension
  the is over.]:
  $
    P (i_b \| t_(b')) = frac(e^(ell_(b b')), sum_b e^(ell_(b b'))) med \, quad P (t_(b') \| i_b) = frac(e^(ell_(b b')), sum_(b') e^(ell_(b b')))
  $

+ Compute the cross-entropy losses in both directions and average:
  $
    cal(L) &= 1 / ( 2B )sum_( b ) (ln P ( i_b|t_b ) + ln P ( t_b|i_b ) ) .
  $<eq_clip_loss>

They also add a temperature to the loss, which they also train.

Post-training, the CLIP models can be used in many ways:

+ Using the vision model as a general purpose feature extractor. This is
  how many vision-language models work: the CLIP image embeddings form
  part of the VLM inputs.

+ Classification works by comparing the logits for a given image across
  embedded sentences of the form `This is an image of a <CLASS HERE>.`

= Mixture of Experts
<mixture-of-experts>
== Basics
<basics>
The $cal(O) ( D ^2 )$ FLOPs count due to MLP
layers#footnote[The $cal(O) ( S
^2 )$ scaling of the self-attention layers is also
untenable, but MoE only addresses the MLP layers.] is untenable past a
given point: inference and training just take too long. Mixture of
Experts#footnote[The original MoE research came out of Google: see
@fedus2022switchtransformersscalingtrillion,
@shazeer2017outrageouslylargeneuralnetworks and related work by these
authors. An excellent MoE paper with open-source every thing is here
@muennighoff2024olmoeopenmixtureofexpertslanguage.] (MoE) models address
this concern by splitting single MLP layer into a number of “expert\"
MLP layers and route a subset of the tokens to a subset of the
experts.\" MoE is a lever for changing the relation between the
per-token FLOPs count and the overall parameter count. Example:
comparing a dense and a MoE model at similar parameter counts, the
expert layer's intermediate dimension is reduced by
$cal(O) ( N _"ex" )$ (the number of experts)
and the FLOPs count is also reduced by this factor. Perhaps
unsurprisingly, MoE experts outperform and train faster than their FLOPs
equivalent dense models (at the cost of more engineering complexity and
a higher memory burden).

The general form of the MoE layer output is
$
  z'_(s d) & = G_(s e) (z_(s d) \, dots.h) E_(e s d)
  (z_(s d))
$<eq_general_moe>
where $G_( s e )(z_( s d ), ... ) in RR^( S times N_"ex" )$ is a gating (i.e.,
weighting) function and $E _( e s d )  ( z _( s d )  ) in RR ^( N _"ex" times S times D )$ is the
usual MLP operation performed by the $e$-th expert. Many of the entries $G_(e s)$ are zero in
practice, and only the computations $E_(e s d) (z_(s d))$ corresponding to non-trivial gating values
are performed, of course. Different MoE variants are essentially differentiated by the specific form
of their weighting function.

== Routing
<routing>
Choosing which experts process which tokens is crucial, affecting both
the downstream model and engineering (i.e. throughput) performance.
There are two dominant schemes:

+ #strong[Token Choice]: each token selects a fixed number of experts.
  $G_(s e)$ is sparse over the expert index; see
  @eq_general_moe;.

+ #strong[Expert Choice]: each expert selects a fixed number of tokens.
  $G_(s e)$ is sparse over the token index; see
  @eq_general_moe;.

Layered on top of this choice are the details of the routing mechanisms.

=== Token Choice vs Expert Choice
<token-choice-vs-expert-choice>
Token and expert choice both introduce a tensor
$W _( d e ) in RR ^( D times N _( "ex"
) )$ which is used to produce a score between each token and expert:
$S_(s e) = z_(s d) W_(d e)$. In each case, we perform a `topk`
computation and output a weighted sum of expert outputs: the two methods
just differ in the dimension over which the `topk` is performed.

For token choice, the gating function is: $
G ^"expert"_( s e )(z _( s d ), W) &= SM_( s )  ( TOPK _( s )  ( z _( s d ) dot W _( d e )  )  )  ,
$<eq_expert_choice> with `topk` acting as in the token choice case.
$G_(s e)$ is sparse along the sequence dimension and has
$N _"ex"k$ non-trivial elements. A (potential) disadvantage
of expert choice is that some tokens may not be routed to any expert at
all, but every expert is at least guaranteed an equal load. In this
case, we effectively have $k = c times  (S )/( N _"ex" )$, with $c$ the capacity factor above.

== MegaBlocks
<megablocks>
The MoE computation maps awkwardly to the typical GPU primitives.
Ideally the expert computations in
@eq_general_moe are parallelized as much
as possible, but
#link("https://pytorch.org/docs/stable/generated/torch.bmm.html")[batched matrix multiplies]
(the closet common primitive) enforces equal token counts per expert,
which is overly restrictive.

MegaBlocks @gale2022megablocksefficientsparsetraining introduces the
proper sparse kernels to handle general MoE computations without the
need to enforce any hard per-expert token limits or introduce
unnecessary padding. They call their method dropless MoE (dMoE).

== MoE Variants
<moe-variants>
A collections of other MoE architecture choices.

=== Shared Experts
<shared-experts>
Shared experts forces one particular expert to always be used, with the
motivation of having the differentiated expert serve as a common pool of
knowledge.

= Inference
<inference>
== Basics and Problems
<basics-and-problems>
The essentials of decoder-only inference is that a given input sequence
$x_(b s)$ is turned into a probability distribution $p_(b s v)$ over the
vocabulary for what the next token might be. Text is then generated by
sampling from $p_(b s v)$ in some way, appending that value to $x_(b s)$
to create a one-token-longer sequence, and then repeating until desired.

There are various problems that naive implementations of the above face:

- Repeated computation from processing the same tokens in the same order
  repeatedly, at least for some sub-slice of $x_(b s)$.

- Inherently sequential computation, rather than parallel

- Sub-optimal sampling strategies. Just choosing the most-probably token
  at each new step, does not guarantee the most-probable overall
  sequence, for instance.

=== Generation Strategies <sec_generation_strats>
A quick tour of generation strategies. A very readable blog post
comparing strategies can be
#link("https://huggingface.co/blog/how-to-generate")[found here.]

==== Greedy <subsec_greedy_gen>
The most obvious generation strategy is to take the final, `(B, S, V)`-shaped outputs $z_(b s v)$
and just take the next token to be the most-probable one (for the final position in the sequence):
`next_token = z[:, -1].argmax(dim=-1)`.

There are various important, practical considerations which are ignored
in the above implementation, including:

- Since we are taking the prediction from the last (`-1`-indexed) element in
  each sequence, it is crucial that all padding is #emph[left]-padding,
  so that these final elements are meaningful.

- Models will signal the end of generation by outputting
  tokenizer-specific codes, and generation must respect these.

See
#link("https://github.com/huggingface/transformers/blob/04ab5605fbb4ef207b10bf2772d88c53fc242e83/src/transformers/generation/utils.py#L1115")[the `generate` method from the `transformers` library]
for more fully-featured code (which, correspondingly, is not always easy
to follow).

==== Simple Sampling: Temperature, Top-$k$, and Top-$p$ <subsec_simple_sampling>
The next-most-obvious strategy is to choose the next token by drawing
from the probability distribution defined by the $z_(b s v)$. There are
various refinements of this idea.

A one-parameter generalization of this strategy introduces a
(physics-motivated) #strong[Temperature] which just adjusts the scale of
the logits:

#block[
  next_token = torch.multinomial((z\[:, -1\] / temp).softmax(dim=-1),
  num_samples=1)

]
assuming `z` are the final logits. Larger temperature yields a larger
variance in the chosen tokens.

With temperature sampling, there is still a non-zero chance of choosing
an extremely improbable token, which is undesirable if you do not trust
the tails of the distribution. Two common truncation strategies which
guard against this:

- #strong[Top-]$k$: Only choose from the top-$k$ most-probable examples
  (re-normalizing the probabilities across those $k$ samples)

- #strong[Top-]$p$: Only choose from the top-however-many most-probable
  examples whose probabilities sum to $p$ (again re-normalizing
  probabilities). This is also some times called #strong[nucleus
  sampling].

==== Beam Search <subsec_beam_search>
Choosing, say, the most-probable next-token at each step is not
guaranteed to yield the most probable #emph[sequence] of tokens. So,
#strong[Beam Search] explores multiple sequences, using different
branching strategies, and the probabilities of the various beam
sequences can be compared at the end. Important note: generating the
most-probable text is not necessarily equal to the most human-like text
@holtzman2020curious.

==== Speculative Decoding <subsec_speculative_decoding>
Speculative decoding @leviathan2023fastinferencetransformersspeculative
is an excellent idea: use a cheaper \"draft\" model to perform the slow,
iterative generation steps and check its work with the full model. Using
a detailed-balance-like construction, it can be guaranteed that this
speculative decoding generation strategy creates text drawn from the
same distribution as the full model.

Informally, the algorithm is:

+ Generate $gamma$ tokens with the draft model, whose distribution is
  $q(x _( t )|x _(
  "prefix" ))$, $t in (0 \, dots.h \, gamma - 1)$. Write
  the generated tokens as $z_t$.

+ Pass the prefix and all $gamma$ generated tokens $z_t$ through the
  full model, which computes probabilities via its distribution
  $p(x _( t )|x _"prefix")$.

+ For every generated token $z_t$, accept it unconditionally if
  $q(z_t| x _"prefix")<= p(x_t| z _"prefix")$.
  If
  $q(z_t| x _"prefix") \> p(z_t| x _"prefix")$,
  instead accept the token with only probability#footnote[This choice is
  not fundamental; it just makes following expressions nicer.] $p / q$.

+ If only the first $n < gamma$ tokens are accepted, generate token
  $n + 1$ from a modified distribution
  $p\'(x_t| x_"prefix") = F \[ p(x), q(x) \]$
  built from the two model predictions and chosen (as will be shown)
  such that the entire algorithm generates the correct distribution.
  $n + 1$ tokens are created in this case#footnote[We cannot generate
  more tokens because drawing from $p'$ effectively changes the prefix
  that the full model should use.].

+ If all of the $gamma$ tokens are accepted, generate token $gamma + 1$
  from the full model's outputs.

Proof of correctness and the derivation of $p' (x)$: let
$Q(x_t| x _"prefix")$ be the distribution described
above. Then this can be broken down according to conditioning on the
draft token and whether or not the draft token was accepted. Dropping
the prefix condition for brevity and $A$ and $R$ stand for rejected and
accepted, respectively, we have $
Q(x_t) &=sum _( z _( t ) ) Q(x_t|z _( t ) , A)P(A|z _( t )) q(z _( t )) + Q(x_t|z _( t ) , R)P(R|z _( t )) q(z _( t )) \
&=sum _( z _( t ) ) delta _( x _( t ), z _( t ) ) times min  (1, ( p(z _( t )) )/( q( z _( t ))
) ) times q (z _( t )) + p\'(x _( t )) (1 - min  (1, ( p(z _( t )) )/( q( z _( t ))
) ) ) times q(z _( t )) \
&= min  (q(x _( t )), p(x _( t )) ) + p\'(x _( t ))sum _( z _( t ) ) (1 - min  (1, ( p(z _( t )) )/( q( z _( t ))
) ) ) times q(z _( t ))
$ The sum is just some constant (denoted by $1 - beta$
in the paper, which should really have a $t$ subscript) and so choosing
$
  p\'(x_( t )| x_"prefix)" &eq.triple ( p(x_( t )| x_"prefix)" - min ( q(x_( t )| x_"prefix)", p(x_( t )| x_"prefix)" ) ) / ( 1- beta )
$ achieves the goal of getting
$Q(x _( t )| x_"prefix)" = p(x _( t )| x_"prefix)"$. It
can be verified that this distribution is properly normalized.

An approximate analysis for choosing the optimal value of $gamma$ can be
found in the paper.

=== The Bare Minimum and the kv-Cache
<sec_kv_cache>

There are two separate stages during generation. First, an original, to-be-continued series of
prompts $x_(b s)$ can be processed in parallel to both generate the first prediction and populate
any intermediate values we may want to cache for later. We follow @pope2022efficiently and call this
the #strong[prefill] stage. For this procedure, we require the entire $x_(b s)$ tensor.

In the second, iterative part of generation (the #strong[decode] stage) we have now appended
one-or-more tokens to the sequence and we again want the next prediction, i.e. `z[:, -1, :]` for the
last-layer outputs $z_(b s d)$. In this stage, we can avoid re-processing the entire $x_(b s)$
tensor and get away with only processing the final, newly added token, #emph[if] we are clever and
cache old results (and accept a very reasonable approximation).

The important pieces occur in the $CA$ layer, as that's the only location in which the sequence index is
not completely parallelized across operations. Referring back to @attn_layer, given the input $z_(b
s d)$ of the $CA$ layer, the re-weighted value vectors#footnote[Summed over $s'$, but concatenating the
  different $a$ values over the $f$ dimension.] $w_(b s s' d)^a v_(b s' f)^a$ are the key objects
which determine the next-token-prediction, which only depends on the $s = - 1$ index values.
Therefore, we can cut out many steps and the minimum requirements are:

- Only the attention weights $w_(b s s' d)^a$ with $s = - 1$ are needed

- The only query values $q_(b s d)^a$ needed to get the above are those
  with $s = - 1$

- Every component of the key and value vectors
  $k_(b s d)^a \, v_(b s d)^a$ is needed, but because of the causal
  mask, all components except for the last in the sequence dimension
  ($s eq.not - 1$) are the same as they were in the last iteration, up
  to a shift by one position#footnote[This is where we need to accept a
  mild approximation, if using a sliding attention window. With an
  infinite context window, if we add a label $t$ which indexes the
  iteration of generation we are on, then we would have that
  $z_(b s d)^((t + 1)) = z_(b (s - 1) d)^((t))$ for every tensor in the
  network, except for when $s = - 1$, the last position. The finiteness
  of the context window makes this statement slightly inaccurate because
  we can only ever keep $K$ positions in context and the loss of the
  early tokens upon sliding the window over will slightly change the
  values in the residual stream.]

So, we are led to the concept of the #strong[kv-cache] in which we cache
old key and query vectors for generation. The cache represents a
tradeoff: fewer FLOPs are needed for inference, but the memory costs are
potentially enormous, since the size of the cache grows with batch size
and sequence length: $
M _"kv-cache" & = 2p B D L S /T ,
$<eq_kv_cache_memory>
in the general case with tensor-parallelism. This can
easily be larger than the memory costs of the model parameter:
$M _"params" ^"inference"  p N _"params"  p D L
^2$ (dropping $cal(O) ( 1 )$ factors), so that the
cache takes up more memory when $B S gt.tilde D$, i.e. when the total
number of token exceeds the hidden dimension. Using the kv-cache
eliminates a would-be $cal(O) ( S ^2 )$ factor in the
FLOPs needed to compute a new token, reducing it to linear-in-$S$
dependence everywhere.

=== Basic Memory, FLOPs, Communication, and Latency
<basic-memory-flops-communication-and-latency>
The essentials of inference-time math, much of it based on
@kipply_inference_math.

==== Naive Inference
<naive-inference>
Processing a single `(B, S, D)`-shaped tensor to generate a single next input costs
the $2B S N _"params"$ FLOPs we found for the forwards-pass
in @sec_flops_training (assuming $S lt.tilde D$). Memory costs just
come from the parameters themselves:
$M _"infer"^"naive"=p N _"params"$. Per
the analysis of App.~@app_compute_mem_bound, naive inference is
generally compute-bound and so the per-token-latency is
approximately#footnote[Assuming we do the naive thing here and generate
the next token in a similarly naive way, shifting over the context
window.] $2B S N _( "params"
)/ lambda _"FLOP/s"$ where the FLOPs bandwidth in the
denominator is again defined in App.~@app_compute_mem_bound.

==== kv-Cache Inference
<kv-cache-inference>
The FLOPs requirements for the hidden-dimension matrix multiplies during
generation are $2 B N _"params"$, since we are only
processing a single token, per previous results. This is in addition to
the up-front cost of $2B S N _(
"params)"$ for the prefill. But, the memory
requirements are raised to $
M _"infer"^"kv-cache" & =p N _"params" + 2 p B D L S/T .
$ Inference now has a computational-intensity of
$
  ( C_"infer"^"kv-cache" ) / ( M_"infer"^"kv-cache" ) & ( B D ) / S ,
$ dropping $cal(O) ( 1 )$ factors, is
now memory-bound (again, see App.~@app_compute_mem_bound), and has
per-token-latency of approximately $M _"infer"/
lambda _"mem"$, unless the batch-size is very large.

==== Intra-Node Communication
<intra-node-communication>
For $T$-way tensor parallelism, two `AllReduce`s are needed, one for each $MLP$ and each $CA$ layer,
where each accelerator is sending $p B D S$ bytes of data (see @subsec_tensor_parallelism). This
requires a total of $4 (T - 1) p B D S \/ T approx 4 p B D S$ bytes to be transferred between
workers in the tensor-parallel group (see @foot_all_reduce), taking a total of $ 4p B D L S/ lambda
_"comms"$ time for the model as a whole. For an A100 80GiB, `torch.bfloat16` setup, this is $ B D S
times 10 ^( -11 )  "sec"$

==== Latency
<latency>
TODO

= Appendix
== Conventions and Notation
<app_conventions>
We loosely follow the conventions of @korthikanti2022reducing. Common
parameters:

- $A$: number of attention heads

- $B$: microbatch size

- $C$: compute (FLOPs)

- $D$: the hidden dimension size

- $E$: expansion factor for MLP layer (usually $E = 4$)

- $H$: $D \/ A$, the head dimension size

- $K$: the block size (maximum sequence length#footnote[In the absence
  of methods such as ALiBi @ALiBi can be used to extend the sequence
  length at inference time.])

- $L$: number of transformer layers

- $N _"params"$: total number of model parameters

- $N _"ex"$: number of experts for MoE models.

- $P$: pipeline parallel size

- $S$: input sequence length

- $T$: tensor parallel size

- $V$: vocabulary size

- $t$: various timescales

- $p$: the precision of the elements of a tensor in bytes

- $lambda$: various rates, e.g. $lambda _"mem"$ is memory
  bandwidth

Where it makes sense, we try to use the lower-case versions of these
characters to denote the corresponding indices on various tensors. For
instance, an input tensor with the above batch size, sequence length,
and vocabulary size would be written as $x_(b s v)$, with
$b in (0 \, dots.h \, B - 1)$, $s in (0 \, dots.h \, S - 1)$, and
$v in (0 \, dots.h \, V - 1)$ in math notation, or as
`x[b, s, v]` in code.

Typical transformers belong to the regime
$ V gt.double D \, S gt.double L \, A gt.double P \, T med . $ For
instance, GPT-2 and GPT-3 @gpt2radford2019language@gpt3brown2020language
have $V
cal(O) ( 10 ^4 )$,
$S, L  cal(O) ( 10 ^3 )$, $L, A  cal(O)
( 10 ^2 )$. We will often assume also assume
that#footnote[This condition ensures that the
$cal(O) ( S ^2 )$ FLOPs cost from self-attention is
negligible compared to $cal(O) ( D ^2 )$
contributions from other matrix multiplies. It should be noted that in
Summer 2023 we are steadily pushing into the regime where this condition
does #emph[not] hold.] $S lt.tilde D$ or the weaker#footnote[This
condition ensures that the cost of reading the
$cal(O) ( D ^2 )$ weights is more than the cost of
reading in the $cal(O) ( B S D )$ entries of the
intermediate representations.] $B S lt.tilde D$.

As indicated above, we use zero-indexing. We also use `python` code
throughout#footnote[Written in a style conducive to latex, e.g. no
type-hints and clarity prioritized over optimization.] and write all ML
code using standard `torch` syntax. To avoid needing to come up with new symbols
in math expressions we will often use expressions like $x arrow.l f (x)$
to refer to performing a computation on some argument ($x$) and
assigning the result right back to the variable $x$ again.

Physicists often joke (half-seriously) that Einstein's greatest
contribution to physics was his summation notation in which index-sums
are implied by the presence of repeated indices and summation symbols
are entirely omitted. For instance, the dot product between two vectors
would be written as
$ arrow(x) dot.op arrow(y) & = sum_i x_i y_i equiv x_i y_i $ We use
similar notation which is further adapted to the common element-wise
deep-learning operations. The general rule is that if a repeated index
appears on one side of an equation, but not the other, then a sum is
implied, but if the same index appears on both sides, then it's an
element-wise operation. The Hadamard-product between two matrices $A$
and $B$ is just $ C_(i j) & = A_(i j) B_(i j) med . $ Einstein notation
also has implementations available for `torch`:
#link("https://rockt.github.io/2018/04/30/einsum")[see this blog post on `einsum`]
or the #link("https://einops.rocks/1-einops-basics/")[`einops`] package. We
strive to write all learnable weights in upper case.

In particular, we use `einops` notation for concatenation and splitting: $A_c = A_((d e)) = B_(d
e)$#footnote[The indexing is all row-major: if $A_i$ is $I$-dimensional, $i in ( 0 \, dots.h \, I -
  1 )$, then if we split this index as $A_i = A_((j k)) equiv macron(A)_(j k)$, then the indices $j
  \, k$ will range over $j in ( 0 \, dots.h \, J )$, $k in ( 0 \, dots.h \, K )$ with $I = J times
  K$ and where numerically $i = j times K + k$. More complex cases follow by induction.]. We will
some times use a bar to indicate tensors which are derived from other tensors through such splitting
operations, usually in the context of tensor-sharding where devices only locally hold some shard of
the tensor. In this context, only some of the dimensions will be sharded across devices, and we may
also put a bar over the corresponding sharded index. For instance, consider a two-dimensional tensor
$M_(a b)$ of shape `M.shape = (A, B)`: sharding this tensor across two devices across the final
index results in a tensor $macron(M)_(a macron(b))$ which is of shape `M_bar.shape=(A, B/2)` on each
device. As here, we will some times use bars to denote indices which are sharded over different
devices.

We also put explicit indices on operators such as $SM$ to help
clarify the relevant dimension, e.g. we would write the softmax
operation over the $b$-index of some batched tensor $x_(b v d dots.h)$
as $
s _( b v d... ) & = ( e^( x _( b v d...) ) )/( sum _( v = 0 ) ^( v= V-1 ) e^( x _(
b v d... ) ) ) eq.triple
SM _( v )  x _( b v d... )
 ,
$<app_eq_einstein_softmax> indicating that the sum over the singled-out
$v$-index is gives unity.

== Collective Communications <app_collective_communications>
A quick refresher on common distributed
#link("https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html")[communication
  primitives]. Consider $R$ ranks with tensor data $x^((r))$ of some arbitrary shape `x.shape`,
which takes up $M$ bytes of memory, where $r$ labels the worker and any indices on the data are
suppressed. For collectives which perform an operation over a specific dimension, the `torch`
convention is that it operates over . The $r = 0$ worker is arbitrarily denoted the #emph[chief].
Some operations are easiest to describe by forming the logical super-tensor `X=torch.stach([x0, x1, ...], dim=0)`
of shape `X.shape=Size(R, ...)` such that the tensor on rank `r` is `x=X[r]`. Then, the primitive operations are:

- `Broadcast`: all workers receive the chief's data, $x^((0))$.

- `Gather`: all workers communicate their data $x_n$ to the chief in a concatenated array
  $[x^0 \, x^1 \, dots.h \, x^(R - 1)]$. E.g., the chief gets `x_out = X.reshape(R*X.shape[1], X.shape[2:])`.

- `Reduce`: data is `Gather`-ed to the chief, which then performs some operation (`sum`, `max`,
  `concatenate`, etc.) producing a new tensor $x'$ on the chief worker. E.g., for `sum` the chief
  gets `x_out = X.sum(dim=0)`.

- `ReduceScatter`: a reducing operation (e.g. `sum`) is applied to the $x^((r))$ to produce a $x'$
  of the same shape (e.g. $x' = sum x^((r))$) and each worker only receives a $1 \/ R$ slice (and
  hence $M \/ R$ byte) of the result#footnote[Note that `AllGather` and `ReduceScatter` are morally conjugate to each other. In
  the former, each worker ends up with $R$ times as much data as they started with, while in `ReduceScatter` they
  end up with $1 \/ R$ of their initial data. One is nearly a time-reversed version of the other,
  which is a way of remembering that they have the came communication cost. They also compose to
  produce an output of the same initial size, as in `AllReduce`.]. A ring implementation sends $M times frac(R
  - 1, R)$ bytes over each link in the ring. E.g., for `sum` rank `r` gets output `x_out = X.sum(dim=0).tensor_split(R, dim=0)[r]`.

- `AllGather`: all data $x^((r))$ is communicated to all workers; each worker ends
  up with the array $[x^0 \, x^1 \, dots.h \, x^(R - 1)]$. Functionally
  equivalent to a `Gather` followed by `Broadcast`. A ring implementation sends
  $M times (R - 1)$ bytes over each link in the ring. E.g., all ranks
  get `x_out = X.reshape(R*X.shape[1], X.shape[2:])`.

- `AllReduce`: all workers receive the same tensor $x'$ produced by operating on
  the $x^((r))$ with `sum`, `mean`, etc. Functionally equivalent to a `Reduce` followed by `Broadcast`,
  or a `ReduceScatter` followed by an `AllGather` (the more efficient choice#footnote[The former
  strategy scales linearly with the number of worker, while the latter
  strategy underlies "ring" `AllReduce` which is (nearly) independent of the number
  of workers: if each worker carries data of size $D$ which is to be `AllReduce`-d,
  a total of $frac(2 (R - 1) D, R)$ elements need to be passed around.
  #link("https://andrew.gibiansky.com/blog/machine-learning/baidu-allreduce/")[See this blog post for a nice visualization]
  or @bandwidthOptimalAllReduce2009 for a relevant
  paper.]<foot_all_reduce>). In the latter case, the total cost is
  $2 M times frac(R - 1, R)$, due to `AllReduce`-ing the initial $M$-sized data,
  and then `AllGather`-ing the $M \/ R$-sized reductions. E.g., for `sum` all ranks get `x_out =
  X.sum(dim=0)`.

- `Scatter`: One worker gives shards of a tensor to all workers. If the worker is
  scattering tensor $T_x$ over the given index, a `Scatter` effectively shards this as $T_x arrow.r
  T_((macron(r) y))$, each worker getting a $macron(r)$-shard. If `x` is the chief's data, rank `r`
  receives `x_out = x.tensor_split(R, dim=0)[r]`.

- `AllToAll`: All workers receive shards of all others worker's tensors. If every
  worker has a tensor $T_(macron(r) y)$, for one value of $macron(r)$,
  which we imagine came from a sharding a tensor
  $T_x = T_((macron(r) y))$, then an over the $y$ index produces
  produces the tensor $T_(z macron(r))$ defined by
  $T_(z macron(r)) = T_x$ on all workers. E.g. rank `r` receives `x_out = X.reshape(X.shape[1], R,
  X.shape[:2])[:,r]`.

== Hardware
<hardware>
Basic information about relevant hardware considerations. Much of the
following is from the
#link("https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html")[NVIDIA docs].

=== NVIDIA GPU Architecture
<nvidia-gpu-architecture>
NVIDIA GPUs consist of some amount of relatively-slow off-chip DRAM
memory#footnote[This is the number usually reported when discussing a
given GPU, e.g. 32GiB for the top-of-the-line A100], relatively-fast
on-chip SRAM, and a number of #strong[streaming multiprocessors] (SMs)
which perform the parallel computations. Inside more-recent GPUs, the
SMs carry both “CUDA cores\" and \"Tensor cores\", where the latter are
used for matrix-mulitiplies and the former for every thing else.

A few numbers of primary importance:

- The rate at which data can be transferred from DRAM to SRAM
  ($lambda _"mem"$)

- The number of FLOP/s, which is more fundamentally computed by
  multiplying the number of SMs by the FLOPS/cycle of each SM for the
  specific operation under consideration (see the NVIDIA docs) by the
  clock rate:
  $N_"SM"dot lambda_"FLOPs/cycle"dot lambda_(
    "clock" )$

The terminology and structure of the memory hierarchy is also important
to understand. Types of memory, from slowest to fastest:

- #strong[Global] memory is the slow, but plentiful, off-chip DRAM. It
  is the type of memory typically used as kernel arguments

- #strong[Constant] memory is read only and accessible by all threads in
  a given block. The size of arrays in constant memory must be known at
  compile time

- #strong[Local Memory] is similarly slow to global memory, but more
  plentiful than register memory, and privately to individual threads
  and is allocated from within a kernel. When registers run out, local
  memory fills the gap

- #strong[Shared] memory is shared between all threads in a given block.
  Shared memory is effectively a user-controlled cache. The size of
  arrays in shared memory must be known at compile time

- #strong[Registers] hold scalar values and small tensors whose values
  are known at compile time. They are local to each thread and they are
  plentiful since each thread needs its own set of registers: 65,536 =
  $2^16$ registers per SM an A100.

#link("https://www.youtube.com/watch?v=QQceTDjA4f4&t=2124s")[An excellent video overview of CUDA and NVIDIA GPU architecture which covers some of the above is here.]

=== CUDA Programming Model
<cuda-programming-model>
The CUDA programming model uses a hierarchy of concepts:

- #strong[Threads] are the fundamental unit of
  execution#footnote[Threads are always physically launched in
  #strong[Warps] which consist of 32 threads.] which each run the same
  CUDA #strong[Kernel], or function, on different data inputs in
  parallel. Threads within the same block (below) may share resources,
  like memory, and may communicate with each other. Individual threads
  are indexed through the #strong[threadIdx] variable, which has
  attributes with in and similar.

- Threads (and hence warps) are organized into 3D #strong[blocks]. The
  size and indices of the blocks can be accessed through the and
  variables, respectively, with in . total threads run in a block.

- Blocks are organized into 3D #strong[groups]. The size of the gird
  dimensions can be accessed through the variable, with similar
  attributes to the above. \
  total blocks run in a grid.

The number of threads which can be launched in a given block is hardware
limited; A100 80GiB GPUs can run up to 1024 threads in a SM at a time
(32 blocks with 32 threads each), for instance. Hence, block and grid
sizes need to be adjusted to match the problem size. There are also
important memory access considerations here. The 1024 threads which can
be launched can also read sequentially from memory and efficient usage
implies that choosing the block size such that we are doing these reads
as often as possible is ideal.

=== NVIDIA GPU Stats <app_gpu_stats>
Summary of some relevant NVIDIA GPU statistics:

#block[
  #figure(
    align(center)[#table(
        columns: 6,
        align: (center, center, center, center, center, center),
        table.header(
          [GPU],
          [Memory],
          [$lambda_"FLOP/s"$],
          [$lambda_"mem"$],
          [$lambda_"math"$],
          [$lambda_"comms"$],
        ),
        table.hline(),
        [#link("https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-nvidia-us-2188504-web.pdf")[A100]], [80GiB], [312
          TFLOP/s], [2.0 TiB/s], [156 FLOPS/B], [300 GiB/s],
        [#link("https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet.pdf")[A100]], [40GiB], [312
          TFLOP/s], [1.6 TiB/s], [195 FLOPS/B], [300 GiB/s],
        [#link("https://images.nvidia.com/content/technologies/volta/pdf/volta-v100-datasheet-update-us-1165301-r5.pdf")[V100]], [32GiB], [130
          TFLOP/s], [1.1 TiB/s], [118 FLOPS/B], [16 GiB/s],
      )],
    kind: table,
  )

]
where

- $lambda _"FLOP/s"$ is flops bandwidth (for `(b)float16` multiply-accumulate ops)

- $lambda _"mem"$ is memory bandwidth

- $lambda_"math" = ( lambda_"FLOP/s" ) / ( lambda_"mem" )$
  is
  #link("https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html#gpu-arch")[math bandwidth]

- $lambda _"comms"$ is one-way communication bandwidth

A useful approximate conversion rate is that
$1  "TFLOP/s" approx 100 "PFLOP/day"$.

Important practical note: the $lambda _"FLOP/s"$ numbers
should be taken as aspirational. Out-of-the box, `bfloat16` matrix-multiplies in `torch`
with well-chosen dimensions tops out around $tilde.op 250$ FLOPS/s

=== Compute-bound vs Memory-bound <app_compute_mem_bound>
If your matrix-multiplies are not sufficiently large on, you are wasting
resources @he2022brrrrfromfirstprinciples. The relevant parameters which
determine sufficiency are $lambda _"FLOP/s"$ and
$lambda _"mem"$, the FLOPs and memory bandwidth,
respectively. The ratio
$lambda _"math" eq.triple ( lambda _"FLOP/s" )/(
lambda _"mem" )$ determines how many FLOPS you
must perform for each byte loaded from memory; see App.~@app_gpu_stats.
If your computations have a FLOPs/B ratio which is larger than
$lambda _"math"$, then you are compute-bound (which is
good, as you're maximizing compute), and otherwise you are
memory(-bandwidth)-bound (which is bad, since your compute capabilities
are idling). The FLOPs/B ratio of your computation is some times called
the #strong[compute intensity] or #strong[arithmetic intensity]. When
compute bound, a process takes time
$ F/lambda _"FLOP/s"$, while memory-bound processes
take time#footnote[Note that the time is not additive, e.g.
compute-bound tasks do not take time
$ F/lambda _( "FLOP/s"
        ) +M/lambda _"mem"$ because they are not
sequential: compute and memory-communications can be concurrent.]
$ M/lambda _"mem"$.

==== Matrix-Multiplications vs. Element-wise Operations
<matrix-multiplications-vs.-element-wise-operations>
For instance, to multiply a `(B, S, D)`-shaped tensor $z_(b s d)$ by a `(D, D)`-shaped
weight-matrix $W_(d d')$, $p (B D S + D^2)$ bytes must be transferred
from DRAM to SRAM at a rate $lambda _"mem"$, after which
we perform $2 B S D^2$ FLOPs, and write the `(B, S, D)`-shaped result back to DRAM
again, for a ratio of $
1/p ( B D S )/( 2B S + D )   ( "FLOPs/B"  )  .
$ We want to compare this against
$lambda _"math"$, which from App.~@app_gpu_stats we take
to be $cal(O) ( 100 "FLOPs/B" )$, and plugging in
any realistic numbers, shows that such matrix-multiplies are essentially
always compute-bound. Compare this to the case of some element-wise
operation applied to the same $z_(b s d)$ tensor whose FLOPs
requirements are $tilde.op C times B D S$ for some constant-factor
$C lt.double S \, D$. Then, then FLOPS-to-bytes ratio is
$tilde.op C / p$, which is #emph[always] memory-bound for realistic
values of $C$. The moral is to try and maximize the number of
matrix-multiplies and remove as many element-wise operations that you
can get away with.

==== Training vs. Inference
<training-vs.-inference>
Finally, we note that the above has implications for the Transformers
architecture as a whole, and in particular it highlights the
difficulties in efficient inference. Under the assumptions of
@sec_flops_training,
$ cal(O) ( B S N _"params" )$ total FLOPs
needed during training, while the number of bytes loaded from and
written to memory are
$cal(O) ( B D L S + N_"params" ) cal(O) ( ( B S N_"params" ) / ( D)+ N_"params" )$
which is $cal(O) ( N _"params" )$ for
not-super-long sequence lengths. The arithmetic intensity is therefore
$cal(O) ( B S )$ and so training is compute-bound in any
usual scenario, even at small $B  cal(O) ( 1 )$
batch sizes (as long as individual operations in the network don't
suffer from outlandish memory-boundedness). The problem during inference
is that (if using the kv-cache; see @sec_kv_cache) we only need to
process a #emph[single] token at a time and so $S arrow.r 1$ in the
numerator in the preceding, while the denominator is also weighed down
by the kv-cache in the attention layers.

In more detail, the $MLP$ layers just process $S = 1$ length tensors during
generation, but are insensitive to the kv-cache, so their intensity
comes from just setting $S = 1$ in the above,
$ tilde.op frac(B D, B + D) med \, $ dropping
$cal(O) ( 1 )$ factors now, while the attention layers
have a ratio of the form
$ tilde.op frac(B D S + B D^2, B D + D^2 + B D S) med \, $ where the
last term in the denominator is due to the cache. Now at small
$B  cal(O) ( 1
)$ batch sizes, both intensities reduce to
$cal(O) ( B )$, which is insufficient to be
compute-bound. In the large $B gt.tilde D \/ S$ limit, they at least
become $cal(O) ( D
)$ and $cal(O) ( 1 + D/S )$,
respectively, which may be enough to be compute-bound, but it's hard to
even get into this regime. Note, the importance of the ratio $D \/ S$.
The hidden dimension fixes the context length scale at which inference
can never be compute-bound, in the absence of additional tricks not
considered here#footnote[One such trick: the multi-query attention of
@subsec_multi_query_attn improves every thing a factor of $A$: the
large batch regime is $B gt.tilde frac(D, A S)$ and the intensity ratio
becomes $cal(O) ( 1 + D/( A S ) )$. An analysis
equivalent to the one performed here can be found in the original paper
@shazeer2019fast.].

=== Intra- and Inter-Node Communication
<intra--and-inter-node-communication>
For intra-node communication, GPUs are connected by either PCIe or
NVLink, generally.

- #link("https://blogs.nvidia.com/blog/2023/03/06/what-is-nvidia-nvlink/")[NVLink]
  interconnects are continually updated and achieve speeds of
  $lambda _"comm" ^(
  "intra" )  300$ GiB/s.

For inter-node communication, nodes are often connected by:

- #link("https://e n.wikipedia.org/wiki/InfiniBand")[InfiniBand]
  apparently also achieves speeds $lambda _"comm" ^(
  "intra" )  100$ GiB/s? Haven't found a
  clear reference. But in any case, the bandwidth is divided amongst the
  GPUs in the node, leading to a reduction by $tilde.op 8$.

== Batch Size, Compute, and Training Time <app_batch_size>
The amount of compute directly determines the training time, but not all
ways of spending compute are equivalent. We follow the discussion in
@mccandlish2018empirical which gives a rule of thumb for determining the
optimal batch size which is some times used in practice. The basic point
is that all of the optimization steps take the gradient $bold(g)$ as an
input, and since the gradient is the average over randomly selected
datapoints, steps are more precise as the batch size increases (with
diminishing returns, past a certain point, but the computational cost
also rises with batch size, and a balance between the two concerns
should be struck.

Consider vanilla SGD and study how the training loss changes with each
step. We randomly sample $B$ datapoints $x in cal(D)$ from the
dataset through some i.i.d. process#footnote[The below uses sampling
with replacement, while in practice we sample without replacement, but
the different is negligible for all practical cases.]. Each
corresponding gradient $bold(g) (x) = partial _( w) cal(L)
 ( w, x  )$ is itself a random variable whose average
is the true gradient across the entire dataset $bar(bold(g))$ and we
take the variance to be $
"Var"\[bold(g) (x), bold(g) (x\')\] & =Sigma
$ for some matrix $Sigma$ with (supressed) indices
spanning the space of model weights. Taking instead the mean of a sum of
such estimates,
$bold(g) _( B )eq.triple 1/B sum _( x in cal(B)) bold(g) (x)$,
the mean stays the same, but the variance reduces in the usual way:
$"Var"\[bold(g) _( B ) (x),
bold(g) _( B )(x\')\]=Sigma /B$.

Study the mean loss across the entire dataset:
$cal(L)  ( w  ) = angle.l cal(L)  ( w, x ) angle.r$. Using SGD we take a step
$w --> w -eta bold(g) _( B )$ and change the loss as
$
  cal(L) ( w -eta bold(g)_( B ) ) = cal(L) ( w ) -eta overline(bold(g))dot
  bold(g)_( B ) + 1 / 2 bold(g)_( B )dot H dot bold(g)_( B ) + cal(O) ( bold(g)_( B )^3 ) ,
$ where $H$ is the true hessian of the loss over the
entire dataset at this value of the weights. Taking the expectation
value and minimizing the results over $eta$ gives the optimal choice:
$
  eta_( star ) & = ( eta_"max" ) / ( 1 + ( B_"noise" ) / B ) , quad eta_"max"eq.triple ( bar(bold(g) )^2 ) / ( bar(bold(g) )dot H dot bar(bold(g) ) ) , quad B_"noise" eq.triple ( TR H dot Sigma ) / ( bar(bold(g) )dot H dot bar(bold(g) )) .
$ Notably, the above supports the usual rule of thumb
that the learning rate should be increased proportionally to the batch
size, at least whenever $B l l B _"noise"$. The
diminishing returns of pushing batch sizes past $B _"noise"$
are also evident. In practice it is too expensive to compute the
Hessian, but thankfully the entirely unjustified approximation in which
the Hessian is multiple of the identity such that $
B _"noise" & approx B _"simple" eq.triple ( TR Sigma )/( bar(bold(g) ) ^2 ) ,
$ is somehow a decent approximation empirically, and an
estimator can be created for $B _"noise"$ in a data-parallel
setup; see @mccandlish2018empirical or
#link("https://github.com/crowsonkb/k-diffusion/blob/ab527a9a6d347f364e3d185ba6d714e22d80cb3c/k_diffusion/gns.py#L1")[Katherine Crowson's implementation]
or
#link("https://github.com/EleutherAI/gpt-neox/blob/408e29d9c746a02d842917bb7447c5c4be0b42d4/megatron/gradient_noise_scale/gradient_noise_scale.py#L1")[neox]
for more.

We can further characterize the trade-off between compute and
optimization steps. The expected decrease in loss per update is then
$
  angle.l delta cal(L) angle.r & approx eta_"max" / ( 1 + B_"noise" / B ) bar(bold(g) )^(2 ) + cal(O) ( eta_"max"^2 ) ,
$ that is, we would need
$1 +  B _"noise" / B $ times as many SGD steps to
make the same progress we would have as compared to full-batch SGD. If
$S _"min"$ is the number of steps that would have been
needed for full-batch SGD, we would need $S=S _"min" + S _(
"min" )  B _"noise" / B $ steps
for minibatch SGD. The total number of examples seen is correspondingly
$E = S _"min" times  ( B _"noise" +B
)eq.triple E _"min"+ S _"min"B$, and so
we see the trade-off between SGD steps $S$ and compute $E$ alluded to
above. These relations can be written as#footnote[The analysis here is
simplified in that it assumes that the noise scale and the chosen batch
size are both time-independent. There is confusing logic treating the
more general case where both $B _"noise"$ and $B$ vary with
step in @mccandlish2018empirical, but in any case, the ultimate
relations they use are effectively the same.] $
 (S / (S _"min" )-1  ) (  E /( E _"min" )-1  ) & =1
$ which represent hyperbolic Pareto frontier curves.
So, solutions are of the form $S= ( alpha
+1  )S _"min"$,
$E= ( 1 / alpha +1 )E_"min"$
and since $E = B S$ the corresponding batch size is
$B _"crit" eq.triple 1 / alpha
B _"noise"$. The parameter $alpha$ characterizes how much
you value the trade-off between these two factors and a reasonable
balance is the $alpha = 1$ solution for which $S = 2S _"min"$, $E=2E _"min"$ and
$B _"crit"= B _"noise"$ exactly.

Correspondingly, in @mccandlish2018empirical they suggest training at
precisely this batch size. But it seems much more relevant to balance
time against compute directly, rather than optimization steps vs
compute. Modeling the total training time by
$T approx S (kappa B + sigma)$ for some $kappa \, sigma$ to model
compute costs#footnote[Computation and communication costs each scale
with $B$, the optimizer step does not (and maybe some overhead?), for
instance.], then the above is equivalent to $
T & = ( ( E _"min" + S _"min" B )  ( kappa B+ sigma  ) / B  .
$ which has a minimum at $
B & = sqrt(( sigma E _"min" )/( kappa S _"min" )) .
$ for which the total time is $
T _"min" & =  ( sqrt(kappa E _"min") - sqrt(sigma S _"min")  ) ^2 .
$ In comparison, the total time for the
$B _"crit" = ( E _"min" )/( S _"min" )$ strategy of @mccandlish2018empirical
gives $T _"min" = 2  ( kappa E
_"min" + sigma S _"min"  )$ which is a
factor of $ 2 /( 1- (
sqrt(sigma kappa B _"noise" ) )/( kappa B _"noise" + sigma ) )$
larger. So, this seems like a better choice of optimal batch size, if
you value your time.

== Initialization, Learning Rates, $mu$-Transfer etc <app_init_lr_mup>
A quick review of common initialization strategies and arguments for
learning rate choices and $mu$-transfer. We follow some mix of
@physicalDL@yang2022tensor@yaida2022metaprincipledfamilyhyperparameterscaling@doshi2023criticalinitializationwidedeep.

The core principles are that, at least at the early stages of training,
we attempt to make identified activations in different blocks have
approximately equal statistics#footnote[Assuming there is some regular
block structure with a corresponding natural identification between
weights and activations in different blocks.] and demand that for each
training step the contribution of each weight's change to the
architecture's outputs should be roughly equal for identified weights in
different blocks. Further, this should occur for all choices of
architectural parameters. In particular large-width, $D arrow.r oo$
limit should be $D$-independent at first non-trivial order, which is the
easiest limit to reason about.

We mostly specialize to very simple cases in the following: MLP-only
models which may have trivial non-linearities.

=== Wide Models are Nearly Gaussian <app_nearly_gaussian_wide_models>
First we discuss the justification of an assumption we make throughout:
the outputs of every block (suitably defined) at initialization are
approximately normally distributed.

Take our model to be $z_i^ell = W_(i j)^ell phi (z_j^(ell - 1))$
where the inputs $z_i^0$ are i.i.d.~Gaussian-normally
distributed#footnote[It may be that these inputs come from some
transformation of other data elements. For example, for an LLM the
$z_i^0$ come from looking up the normally-distributed embedding vectors
corresponding to the relevant tokens in the sequence.]:
$E( z^0_( i )) = 0$, $E( z ^0_( i )z ^0_( j ))
= delta _( i j )$. Here, $i in (0 \, dots.h \, D - 1)$ and the batch
and any other indices are suppressed.

Examine the statistics of the first layer. Choosing the weights to be
normally distributed as well, with $E( W^( ell )_( i j )) = 0$,
$E( W ^( ell )_( i j )W ^( ell )_( j k
)) = ( C _( ell ) )/( D ) delta _( i j )$ for some $C_ell$ it
straightforward to show that $
E( z^1_( i )) &= 0 \
E( z^1_( i )z^1_( j )) &= C _( 1 )delta _( i j ) angle.l phi(z )^2angle.r
$ where
$angle.l phi(z)^( n )angle.r eq.triple integral dif thin rho(z) phi(z)^( n )$
with $rho (z)$ a single-variable standard normal Gaussian#footnote[This
is similar notation as used in @physicalDL, with $E(dot)$ being a
multivariate expectation value and $angle.l dot.op angle.r$ an
expectation value over a 1D distribution.] (the $D$ in the denominator
was chosen to counteract a factor of $D$ from an index sum), which are
all some $cal(O) ( 1 )$, $D$-independent numbers.

The first two moments can therefore be made Gaussian-normal-like by
choosing $C _( 1 ) = 1/ E( phi(z)phi(z ))$. Since this can
always be done, the first non-trivial test of non-Gaussianity is the
four-point function (the three-point function vanishes by symmetries).
The connected four-point function#footnote[Also known as the cumulant:
$E( z^( ell )_( i )z^( ell )_( j )z^( ell )_(
k )z^( ell )_( l)) _( c ) eq.triple E( z^( ell )_( i )z^( ell )_( j )z^( ell )_(
k )z^( ell )_( l)) - E( z^( ell )_( i )z^( ell )_( j ))E( z^( ell )_(
k )z^( ell )_( l)) - "perms"$.] show the presence of
non-gaussianity most directly. Symmetries fix the result to be of the
form $
E( z^( ell )_( i )z^( ell )_( j )z^( ell )_(
k )z^( ell )_( l)) _( c ) &= V ^( ell ) _( 4 )delta _( i j )delta _( k l ) + "perms"  ,
$ for some coefficient $V_4^ell$ for all $ell$. We can
fix the coefficient by computing the term, say, where
$i = j \, k = l \, i eq.not k$. The result for the $ell = 1$ layer is:
$
  V^(ell = 1 )_( 4 ) &= ( C_( 1 )^2 ) / ( D^2 ) \[ E( ( phi(z ^0))dot phi(z ^0)) )^2 ) - ( E( phi(z ^0))dot phi(z ^0))) )^2 \]
$ where the expectation is over the distribution of
$z_i^0$ and $W_(i j)^1$ and the dot-product is over hidden-dimension
indices. This can be written in terms of the single-variable expectation
values $angle.l phi (z)^n angle.r$ with the result:
$
  V_4^(ell = 1) & = C_1^2 / D (angle.l phi (z)^4 angle.r - angle.l phi (z)^2 angle.r^2) med .
$
So, there is indeed non-gaussianity (even for a linear network
$phi (x) = x$) and is it of $cal(O)
( 1 / D  ) l l 1$. Perhaps unsurprisingly,
we can continue on to deeper layers via perturbation theory and find
$V ^( ell ) _( 4 )  cal(O) (  ell / D
)$; the non-linearity is additive in the $L \/ D lt.double 1$
regime. Similar results also hold for higher-order, even-point
functions.

We will assume that arguments like this can be generalized for all
networks under consideration: many activations are approximately
Gaussian-normally distributed, after appropriately tuning initialization
scales. Demonstrating this rigorously is a central goal of the Tensor
Programs work @yang2022tensor.

=== muTransfer and Similar Ideas
<mutransfer-and-similar-ideas>
muTransfer @yang2022tensor and similar work in
@physicalDL@yaida2022metaprincipledfamilyhyperparameterscaling@doshi2023criticalinitializationwidedeep
study reasonable prescriptions for how to initialize weights and set
learning rates in a natural way. A practical consequence of these ideas
is that they tend to correspond to families of models, related to each
other by rescalings of architectural hyperparameters, and the optimal
learning algorithm hyperparameters (such as the learning rate) for
members of the family tend to be similar, much more so than is found for
alternative schemes.

We start by working through the general criteria for an extremely simple
model, and then see how it extends to more general cases.

==== A Toy Limit: Deep Linear Networks and SGD<app_mup_toy_limit>
Take the model to be very simple: a deep linear model without any
biases. Though simple, the conclusions we reach for this model will
essentially all carry over to more complex cases. Whatever general
prescription we come up with should work in this limit, at the very
least.

The model is: $
z _( o ) eq.triple z ^( L )_( o ) &= O _( o d ) H ^( L-1 )_( d d\' ) ... H ^0 _( d\'\' i )I _( i \'i ) x _( i ) \
z ^( ell )_( d ) & eq.triple H ^( ell )_( d d\' ) z ^( ell-1 )_( d\' )  , ell in  ( 0, ... , L-1  )\
z ^(-1)_( i ) & eq.triple I _( i i\' ) x _( i\' )  .
$<app_eq_deep_linear_model>
Typically, $ell$ is only used to index the hidden
layers, and we suppress any batch or sequence dimensions for simplicity.
The $H_(d d')^ell in bb(R)^(D_ell times D_(ell - 1))$ are the hidden
layer weights, such that $z_d^ell in bb(R)^(D_ell)$,
$O_(o d) in bb(R)^(D_O times D_(L - 1))$ is the readout layer (e.g. LM
head for a LLM), and $I_(d i) in bb(R)^(D_(- 1) times D_I)$ the input
layer (e.g. embedding layer for a LLM). The $D_ell$ dimensions may all
be different.

We will consider a family of models which are related by expanding all
of the hidden dimensions, which is the easiest limit to analyze:
$
  D_ell arrow.r lambda D_ell med \, quad D_(- 1) arrow.r lambda D_(- 1) med \, quad D_O arrow.r D_O med \, quad D_I arrow.r D_I med .
$<app_eq_mup_width_scaling>
The input an output dimensions are not scaled, since these are typically
fixed by the problem at hand (e.g., both are the vocab size for an LLM).

Our goal is to choose the hyperparameters (weight initialization, learning rates, etc.) such that
all models in this class have similar behaviors, which translates to the model outputs and updates
all being $lambda$ independent at first non-trivial order, in a way made more precise below. We
require#footnote[In general, we define the size of a random variable $Z$ through the size of its
first non-vanishing moment: if it's the $n$-th moment, then we write $Z ~  cal(O) ( angle.l Z^( n)
angle.r^(1/n) )$. The common cases are when the $Z$ has a non-trivial mean, $Z~ cal(O) ( angle.l Z
angle.r )$, and the case where the mean is zero, but the second moment is non-trivial: $Z ~ cal(O)
( sqrt(angle.l Z Z angle.r) )$.]:

- All intermediate tensors
  $z ^( ell ) _(d )~  cal(O) ( 1 )$.

- Model outputs are
  $z^( ell )_( d ) ~cal(O) ( 1 )$ #emph[after]
  taking an optimizer step.

These requirements fix the scaling of every parameter of interest with
respect to $lambda$, with a single degree of freedom remaining.

Assume that the $x _(i )  cal(O) ( 1 )$, either by
whitening or because they're one-hot ($x_i = delta_(i v)$, for some
$v$), as in the LLM case. Then for the #strong[base model], defined to
be the one with $lambda = 1$, we already know how to achieve the first
criteria above. Let the input weight components be chosen so that they
generate independent, normally distributed outputs. We consider two
scenarios:

+ If the inputs are approximately normally distributed (say be whitening
  features), then we take
  $angle.l I_(d i) I_(d' i') angle.r = frac(delta_(d d') delta_(i i'), D_I)$.

+ If the inputs are instead one-hot (as in LLMs), then we take
  $angle.l I_(d i) I_(d' i') angle.r = delta_(d d') delta_(i i')$.

Both scenarios produce outputs, defined to be $z_d^(- 1)$, which
obey#footnote[We consider the inputs $x_i$ fixed, for simplicity. A
better treatment would also consider the data distribution.]:
$
  angle.l z_d^(- 1) angle.r = 0 thin quad angle.l z_d^(- 1) z_(d')^(- 1) angle.r = delta_(d d') med \,
$
with $angle.l dot.op angle.r$ an expectation value over all weight
distributions. Subsequently, all of the $z_d^ell$ will be zero mean with
unit two-point correlation functions#footnote[They are not normally
distributed, however: their higher-point, connected correlation
functions are non-trivial @physicalDL. This is expected, since the
$z^ell$ for $ell gt.eq 0$ are products of Gaussian random variables, and
such a product is not Gaussian. However, the degree of non-Gaussianity
is small: $cal(O) ( ell/D )$.] if we
initialize the $H_(d d')^ell$ as
$
  angle.l H_(d e)^ell H_(d' e')^ell angle.r & = frac(delta_(d d') delta_(e e'), D_(ell - 1)) arrow.r.double.long angle.l z_d^ell angle.r = 0 med \, quad angle.l z_d^ell z_(d')^ell angle.r = delta_(d d') med .
$
We will leave the variance of the output layer undetermined for now:
$
  angle.l O_(o d) O_(o' d') angle.r & = frac(delta_(o o') delta_(d d'), D_(L - 1)^(1 + s)) med \, quad arrow.r.double.long ⟨z_d^L⟩ = 0 med \, quad ⟨z_d^L z_(d')^L⟩ = delta_(d d') D_(L - 1)^(- s) med \,
$
for some $s$ (chosen to match the $s$ of
@yaida2022metaprincipledfamilyhyperparameterscaling. Setting $s = 0$
would yield $D$-independent, order-one model outputs at initialization,
$z^( L ) cal(O) ( 1 )$, but we'l l see that this is
not the only viable choice (and muTransfer uses $s = 1$).

Now take an optimization step due to the loss from, say, a single input
$x_i$. We allow for per-weight learning rates
$(eta_I \, eta_ell \, eta_O)$, for the input, hidden, and output
weights, respectively. Taking a step and computing the model outputs on
an input $y_i$ (possibly different from $x_i$), the updated model's
outputs $z^L (y_i \, t = 1)$ are related to the value it would have had
at initialization, $z^L (y_i \, t = 0) equiv z^L (y_i)$, via
$
  z^( L )_( o ) ( y, t=1 ) &= z^( L )_( o ) ( y )
  + ( partial z^( L )_( o )(y) ) / ( partial I_( d i ) ) Delta I_( d i )(x)
  + ( partial z^( L )_( o )(y) ) / ( partial H^( ell )_( o d ) ) Delta H^( ell )_( o d )(x)
  + ( partial z^( L )_( o )(y) ) / ( partial O_( o d ) ) Delta O_( o d )(x)+cal(O) ( Delta X^2 ) \
  &= z^( L )_( o ) ( y )\
  & - ( partial cal(L)(z(x)) ) / (partial z^( L )_( o\' ) )( eta_( I ) ( partial z^( L )_( o\' )(x) ) / ( partial I_( d i ) )( partial z^( L )_( o )(y) ) / ( partial I_( d i ) )
    + eta_( ell )( partial z^( L )_( o\' )(x) ) / ( partial H^( ell )_( d d\' ) )( partial z^( L )_( o )(y) ) / ( partial H^( ell )_( d d\' ) )
    + eta_( O )( partial z^( L )_( o\' )(x) ) / ( partial O_( o\'\'d ) )( partial z^( L )_( o )(y) ) / ( partial O_( o\'\'d ) ) ) \
  & quad+cal(O) ( eta^2 ) ,
$<app_eq_general_output_update> sum over $ell$ and all other repeated indices
implicit. The above uses SGD with per-weight learning rates, with the
final line obtained after specializing weight updates
$W arrow.r W + Delta W$ to SGD#footnote[The term in parentheses is the
neural tangent kernel, a fundamental quantity which characterizes how
the network gets updated.].

We are interested in the typical size of the updates in @app_eq_general_output_update, for which we
compute the following expectation values:
$
  angle.l ( partial z^( L )_( o\' )(x) ) / ( partial I_( d i ) )( partial z^( L )_( o )(y) ) / ( partial I_( d i ) ) angle.r &= angle.l O_( o\'d\' )H^( L-1 )_(d\'e\' ) ... H^0_( f\'d ) times O_( o d\'\' )H^( L-1 )_(d\'\'e\'\' ) ... H^0_( f\'\'d ) angle.r x_( i ) y_( i )\
  &= delta_( o o' )( x dot y ) / ( D^( s )_( L-1 ) ) \
  angle.l ( partial z^( L )_( o\' )(x) ) / ( partial H^( ell )_( d d\' ) )( partial z^( L )_( o )(y) ) / ( partial H^( ell )_( d d\' ) ) angle.r &= angle.l O_( o\'e\' )H^( L-1 )_(e\'f\' ) ... H^( ell+1 )_( g\'d ) z^( ell-1 )_( d\' ) times O_( o e )H^( L-1 )_(e f ) ... H^( ell+1 )_( g d ) z^( ell-1 )_( d\' ) angle.r \
  &= delta_( o o' ) ( angle.l z^( ell-1 )(x)dot z^( ell-1 )(y) angle.r ) / ( D^( s )_( L-1 ) )\
  &= delta_( o o' ) (D_( ell-1 )) / ( D^( s )_( L-1 ) ) times cases( x dot y wide & x "and" y  "one-hot", ( x dot y )/( D_( I ) ) & x "and" y  "normal")\
  angle.l ( partial z^( L )_( o\' )(x) ) / ( partial O_( o\'\'d ) )( partial z^( L )_( o )(y) ) / ( partial O_( o\'\'d ) ) angle.r &= delta_( o\'o\'\'\' )delta_( o o'\' ) angle.l z^( L-1 )(x)dot z^( L-1 )(y) angle.r \
  &= delta_( o o' ) D_( L-1 ) times cases( x dot y wide & x "and" y  "one-hot", ( x dot y )/( D_( I ) ) & x "and" y  "normal") .
$<app_eq_mup_expectation_vals>

The above are useful if we can compute the expectation value of the model output updates, $Delta
z_o^L equiv z_o^L (t = 1) - z_o^L (t = 0)$, @app_eq_general_output_update as in
$
  angle.l Delta z^( L )_( o ) ( y ) angle.r &approx
  - angle.l ( partial cal(L)(z(x)) ) / (partial z^( L )_( o\' ) )( eta_( I ) ( partial z^( L )_( o\'
    )(x) ) / ( partial I_( d i ) )( partial z^( L )_( o )(y) ) / ( partial I_( d i ) )
    + eta_( ell )( partial z^( L )_( o\' )(x) ) / ( partial H^( ell )_( d d\' ) )( partial z^( L )_( o )(y) ) / ( partial H^( ell )_( d d\' ) )
    + eta_( O )( partial z^( L )_( o\' )(x) ) / ( partial O_( o\'\'d ) )( partial z^( L )_( o )(y) ) / ( partial O_( o\'\'d ) ) ) angle.r\
  &approx
  - angle.l ( partial cal(L)(z(x)) ) / (partial z^( L )_( o\' ) )angle.r angle.l( eta_( I ) ( partial
  z ^( L )_( o\' )(x) )/( partial I_( d i ) )( partial z^( L )_( o )(y) )/( partial I_( d i ) )
+ eta_( ell )( partial z ^( L )_( o\' )(x) )/( partial H^( ell )_( d d\' ) )( partial z^( L )_( o )(y) )/( partial H^( ell )_( d d\' ) )
+ eta_( O )( partial z ^( L )_( o\' )(x) )/( partial O_( o\'\'d ) )( partial z^( L )_( o )(y) )/( partial O_( o\'\'d ) )  ) angle.r .
$

This questionable assumption, which appears to be made in @yang2022tensor as well as @physicalDL, is
at least justifiable in the limit of a mean-squared-error loss or in the essentially-equivalent
limit where the loss is well-approximated as a quadratic expansion about its minimum. Using
@app_eq_mup_expectation_vals with this assumption gives
$
  angle.l Delta z_o^L (y) angle.r & approx - angle.l frac(partial cal(L) (z (x)), partial z_o^L)angle.r times (eta_I x dot.op y + eta_ell D_(ell - 1) + eta_O D_(L - 1)^(1 + s)) / D_(L - 1)^s med .
$

Some conclusions from
/* @app_eq_deltaz_scaling;: */

- Assuming all the $D_ell$ are of roughly the same size, collectively
  called $D$, it is fairly natural to use per-layer learning rates of
  the form
  $
    eta_I & = eta D^s med \, quad eta_ell = eta D^(s - 1) med \, quad eta_O = eta / D med \,
  $
  for some common global learning rate hyperparameter $eta$,
  say#footnote[More generally, these should be taken as individually
  tunable, $D$-independent hyperparameters.]. This ensures that the
  updates from each parameter contributes the same
  $cal(O) ( eta
  )$ (and $D$-independent) shift to the model's outputs.
  With this choice, the model updates will be $lambda$-independent under
  the scaling
  /* @app_eq_mup_width_scaling;, */
  at the current order of approximation. Note that it's not at all
  obvious whether such a stability condition also implies optimal
  learning.

- Equivalently, one could imagine performing a scan over the space
  $(eta_I \, eta_ell \, eta_O)$ to find the optimal learning rates for
  fixed model widths $(D_I \, D_ell \, D_O)$. Then, one should be able
  to scale up the model as in
  /* @app_eq_mup_width_scaling */
  while simultaneously scaling
  $
    eta_I & arrow.r eta_I lambda^s med \, quad eta_ell arrow.r eta_ell lambda^(s - 1) med \, quad eta_O arrow.r eta_O / lambda med \,
  $<app_eq_mup_lr_lambda_scaling_sgd>
  and retain nearly-optimal training. This is closer to the presentation
  in @yang2022tensor.

- The parameter $s$ is currently undetermined. The muTransfer limit
  @yang2022tensor corresponds
  /* to#footnote[@app_eq_mup_lr_lambda_scaling_adam */
  agrees with Table 3 of @yang2022tensor when $s = 1$.] $s = 1$, for
which the model outputs at initialization scale as
$z_d^L tilde.op 1 / sqrt(D)$, an undesirable scaling which is
compensated for by the $D$-independent SGD updates which eventually
make the model outputs approximately independent of model width.

- One reasonable constraint is $s gt.eq 0$, since the model outputs at
  initialization are $z_d^L tilde.op D^(- s \/ 2)$ and we want the
  $Delta z^( L ) cal(O) ( eta
  )$ SGD updates to remain non-trivial in the
  $D arrow.r oo$ limit.

An essentially-equivalent, but differently presented, line of inquiry
comes from @yaida2022metaprincipledfamilyhyperparameterscaling in which
they consider the so-called $ell$-th layer neural tangent
kernel#footnote[They use $H$ where we use $N$ to denote the kernel]:
$
  N_(d d')^ell (y \, x) & = eta_I frac(partial z_(d')^ell (x), partial I_(e i)) frac(partial z_d^ell (y), partial I_(e i)) + eta_ell frac(partial z_(d')^ell (x), partial H_(e f)^ell) frac(partial z_d^ell (y), partial H_(e f)^ell) + eta_O frac(partial z_(d')^ell (x), partial O_(o f)) frac(partial z_d^ell (y), partial O_(o f)) med \,
$
where various terms are zero depending on the value of $ell$ and which
coincides with the full neural tangent kernel when $ell = L$. These obey
recursion relations, from the chain rule: $
N^( L )_( o o' )(y, x) &=eta_( O )delta_( o o' ) z^( L-1 )(x)dot z^( L-1 )(y) + O_( o d )O_( o\'d\' )N^( L-1 )_( d d\' )\
N^( ell )_( d d\' )(y, x) &=eta_( ell )delta_( d d\' ) z^( ell-1 )(x)dot z^( ell-1 )(y) + H^( ell )_( d e )H^(ell )_( d\'e\' )N^( ell-1 )_( e e' )  , quad L-1 g e ell g e 1\
N^(0)_( d d\' )(y, x) &=eta_(0)delta_( d d\' ) z^( -1 )(x)dot z^( -1 )(y) + I_( d i )I_( d\'i\' ) N^( -1 )_( i i' )\
N^( -1 )_( d d\' )(y, x)&=eta_( I )delta_( d d\') x dot y  .
$ Demanding that the true neural tangent kernel
$N_(o o')^L$ be width-independent and that all layers provide
parametrically-equal contributions lands us on the same equations and
solutions as above#footnote[The equivalence follows from the fact that
$angle.l Delta z^( L )_( o )
angle.r = -  angle.l ( partial cal(L)  )/( partial o\' ) N^( L )_( o o' )
angle.r$]. Extending this analysis to higher orders in $eta$,
@yaida2022metaprincipledfamilyhyperparameterscaling derives another
bound: $s lt.eq 1$, placing muTransfer's prescription at the edge of the
bounded region.

===== Adam
<adam>
Since Adam(W), not SGD, is the d e facto optimizer of choice, we need to
extend the above arguments. A quick and dirty assumption which leads to
a reasonable (and phenomenologically supported) scaling result is to
assume that the SGD and Adam updates generally point in the same
direction and that elements of the Adam updates are all
$cal(O) ( 1 )$.

That is, let the Adam update for some weight
$W_(d e) in RR^(D times E)$ be, schematically,
$
  Delta^"Adam" W_( d e ) =- ( angle.l ( partial cal(L) ) / ( partial W_( d e ) ) angle.r ) / ( sqrt( angle.l  (( partial cal(L) )/( partial W_( d e ) ) )^2 angle.r ) ) ,
$ while
$Delta^"Adam" W_( d e ) =- ( partial cal(L) )/( partial W_( d e ) )$,
omitting the learning rate and where expectation values are really
corrected exponential moving averages. Then assume that a relation of
the form $Delta^"Adam" W_( d e )approx
alpha_( W ) Delta^"SGD" W_( d e )$ holds for some
$alpha_W$, i.e. that the Adam and SGD updates typically point in the
same direction and are approximately related by an overall
factor#footnote[This is exactly true in the $beta_1 arrow.r 0$ limit,
using the
#link("https://pytorch.org/docs/stable/generated/torch.optim.Adam.html")[`torch` parameterization]
of Adam (which is the opposite of the usual regime).], such that this
replacement can be made in any expectation value while only inducing
small errors (whose size we will not attempt to quantify).

With these assumptions, we can determine the value of $alpha_W$ through
$
  angle.l Delta^"Adam" W_( d e )Delta^"Adam" W_( d e ) angle.r
  &approx alpha_( W )^2 angle.l Delta^"SGD" W_( d e )Delta^"SGD" W_( d e ) angle.r .
$

The left hand side is approximately $D times E$ (the
number of components of the weight, more generally) via the assumption
that all components $Delta^"Adam" W_( d e )$ are
$cal(O)
( 1 )$. For instance, this is exactly true in the limit
where every gradient in the history has taken on the same value, in
which case all components#footnote[Ignoring the $epsilon.alt$ term in
the Adam implementation. This result is also exactly true for the LION
optimizer @chen2023symbolicdiscoveryoptimizationalgorithms for which
$Delta^"LION"    W_( d e ) = plus.minus 1$ for all components.] are 1. The right side can be approximated
using the same assumptions used in App.~@app_mup_toy_limit:
$
  angle.l Delta^"SGD" W_( d e )Delta^"SGD" W_( d e ) angle.r &= angle.l ( partial cal(L) ) / ( partial W_( d e ) )( partial cal(L) ) / ( partial W_( d e ) ) angle.r\
  &=angle.l ( partial cal(L) ) / ( partial z_( o ) )( partial z_( o ) ) / ( partial W_( d e ) )(partial cal(L) ) / ( partial z_( o\' ) )( partial z_( o\' ) ) / ( partial W_( d e ) ) angle.r\
  &approx angle.l ( partial cal(L) ) / ( partial z_( o ) )(partial cal(L) ) / ( partial z_( o\' ) )angle.r angle.l ( partial z_( o ) ) / ( partial W_( d e ) )( partial z_( o\' ) ) / ( partial W_( d e ) ) angle.r .
$

These final factors were computed in @app_eq_mup_expectation_vals and the relations for the various
weights become:
$
  D_( -1 )D_( I ) &=alpha_( I )^2 angle.l ( partial cal(L) ) / ( partial z_( o ) )(partial cal(L) ) / ( partial z_( o ) )angle.r ( x dot y ) / ( D^( s )_( L-1 ) )\
  D_( ell )D_( ell-1 ) &=alpha_( ell )^2 angle.l ( partial cal(L) ) / ( partial z_( o ) )(partial cal(L) ) / ( partial z_( o ) )angle.r ( D_( ell-1 ) ) / ( D_( L-1 )^( s ) ) times cases( x dot y & x, y  "one-hot" ( x dot y )/( D_( I ) ) & x, y  "normal")\
  D_( O )D_( L-1 ) &=alpha_(O)^2 angle.l ( partial cal(L) ) / ( partial z_( o ) )(partial cal(L) ) / ( partial z_( o ) ) angle.r D_( L-1) times cases( x dot y & x, y  "one-hot" ( x dot y )/( D_( I ) ) & x, y  "normal")\
$
Considering the scaling @app_eq_mup_width_scaling;, we assume the $angle.l ( partial cal(L) )/(
partial z_( o ) )(partial cal(L) )/( partial z_( o ) ) angle.r$ factors to be $lambda$-independent
(since the goal of muTransfer is to keep the model outputs $z_0^L$ $lambda$-independent) and
matching the remaining factors gives:
$
  alpha_I prop alpha_ell prop lambda^(frac(1 + s, 2)) med \, quad alpha_O prop 1 med .
$

Repeating the analysis of App.~@app_mup_toy_limit with the Adam updates and the preceding
assumptions amounts to replacing $eta_X arrow.r eta_X alpha_X$ everywhere, which changes
@app_eq_mup_lr_lambda_scaling_sgd to#footnote[@app_eq_mup_lr_lambda_scaling_adam agrees with Table 3
  of @yang2022tensor when $s = 1$.]
$
  eta_( I ) &--> eta_( I ) lambda^( ( s-1 ) / ( 2 ) ) ,quad
  eta_( ell ) --> eta_( ell ) lambda^( ( s-3 ) / ( 2 ) ) , quad
  eta_( O ) --> eta_( O ) / ( lambda ) quad ("Adam)" .
$<app_eq_mup_lr_lambda_scaling_adam>

===== Activations
<activations-1>
Adding in non-trivial activation functions, such that
$z_d^ell = W_(d d')^ell phi (z_(d')^(ell - 1))$, does not
qualitatively change the picture. The two relatively minor changes are:

- The conditions for maintaining
  $z^( ell ) cal(O) ( 1 )$ in the forwards pass
  are slightly altered, by $cal(O) ( 1 )$
  factors#footnote[E.g. for $mono("relu")$ we could double the variance
  to keep unit covariance of the intermediates:
  $⟨W_(d e)^ell W_(d' e')^ell⟩ = 2 / D_(ell - 1) delta_(d d') delta_(e e') arrow.r.double.long ⟨z_d^ell z_(d')^ell⟩ = delta_(d d')$.].

- The chain rule factors which were previously
  $frac(partial z_d^ell, partial z_(d')^(ell - 1)) = W_(d d')^ell$ now
  turn into
  $frac(partial z_d^ell, partial z_(d')^(ell - 1)) = W_(d d')^ell phi' (z_(d')^(ell - 1))$.

Both of these affect scaling with depth, $L$, but not the scaling with width
@app_eq_mup_width_scaling in any essential way.

== Cheat Sheet <app_cheat_sheet>
Collecting all of the most fundamental equations, given to various
degrees of accuracy.

Number of model parameters: $
N _"params" & = (4+2E)L D ^2 + V D+ cal(O) ( D L ) approx  ( 4+2E  )L D ^2 ,
$ assuming no sharding of the embedding matrix.

=== Training
<training-1>
Memory costs for mixed-precision training: $
M _"model" & =p _"model" N _"params" \
M _"optim" & =  ( s _"states"+1 ) times p _"master" N _"params" \
M _"act" ^"total" & =( 2B D L S  ( p(E+4) + 1  ) )/( T )
+ ( A B L S ^2  ( 2p+1 ) )/( T ) + cal(O) ( B S V )
$ where $s _"states"$ is the number of
optimizer states, e.g. $s = 0$ for SGD and $s = 2$ for Adam. FLOPs
total: $
F _"total" ^"model" & approx 12 B D L S  ( S +  ( 2+E  )D  ) .
$

#bibliography("bibliography.bib")
