# Number representations in transformers

Research notes, 2026-10-16.

The current design question is a **fixed-capacity dictionary of concrete
rational values, with membership selected during training**. See
[Learning a finite vocabulary of rational numbers](learned-rational-vocabulary.md)
for the proposal, exact handling of off-dictionary values, and the separate
open question about learned components.

The review below also examines the earlier, stronger hypothesis of **an
independently trainable vector for each complete numerical value**, together
with a categorical output. A complete value can be a fraction, not just an
integer. References to the strong hypothesis mean that broader requirement;
the current fixed-capacity proposal gives independent rows to selected values.

The deeper review identifies three useful directions:

- **Preserve independent rows:** numerical graph embeddings such as
  [NEKG](#nekg-learned-vectors-for-numerical-entities), combined with
  [dynamic embedding storage](#dynamicembedding-growing-the-table), offer
  precedents for learning and extending a finite numerical table.
- **Improve their training:**
  [GrokTransfer and embedding-specific optimization](#what-the-grokking-papers-establish)
  directly study the role of embeddings in delayed generalization.
- **Change the representation if the premise is relaxed:**
  [FoNE and BitTokens](#recent-approaches-to-one-vector-per-number) encode a
  complete value with shared rules. A separate
  [modular reconstruction proposal](#a-conditional-route-from-finite-vocabularies-to-exact-fractions)
  connects finite residue vocabularies to exact bounded fractions.

These are separate results and proposals, not a published combined solution.
In the sources reviewed here, I did not find a demonstration combining
independent vectors for an expanding rational vocabulary, exact rational
input/output, and grokking on previously unseen numerical identities.

The [local scalar MLP report](grokking-results/div97-mlp/report.md) motivates
the hypothesis: several configurations fit the training data well while
held-out accuracy remained weak. The experiments did not separately establish
which changes to representation, output, architecture or optimization were
necessary.

## Independent vectors and generated vectors

An independent lookup table has the form

$$
e(x)=E[\operatorname{id}(x)],\qquad E\in\mathbb{R}^{V\times d}.
$$

Each supported value has its own row of parameters. Parameter sharing in the
rest of the model does not remove that row's freedom to change independently.

A generated embedding instead has the form $e(x)=f_\theta(x)$, or
$e(x)=f_\theta(\operatorname{characters}(x))$. The same parameters $\theta$
produce vectors for many numbers. This can accept a previously unseen number
without allocating a new row, but changes the strong hypothesis above.

Four properties need separate checks: one token position per value; independent
parameters per value; a lossless representation over a stated domain; and an
output mechanism that can express the answer. A method can satisfy some of
these without satisfying all four. Likewise, categorical prediction of each
digit or bit differs from one output class per complete value.

| Approach | What has independent trainable parameters? | Treatment of a new number |
| --- | --- | --- |
| Exact lookup table | Each supported complete value | Requires a new row and training |
| FP15 | Each supported rounded value | Rounds into a finite vocabulary |
| Chronos | Each quantization token | Scales and assigns it to a bin |
| Prototype embeddings | A finite set of numerical prototypes | Mixes existing prototype vectors |
| LUNA / NumBed | A shared encoder of number strings | Computes one vector from its characters |
| NumGPT | Exponent embeddings and shared model parameters | Combines exponent and mantissa features |
| DICE | Base geometry is prescribed; contextual adaptation can be learned | Applies the geometric encoding within its configured range |
| Numerical feature embeddings | A shared function per input feature | Evaluates the function at the new value |
| xVal | One shared numerical direction in the basic scheme | Scales that direction by the value |
| Digit tokens with Abacus | Digit embeddings and positional embeddings | Represents the number as a longer sequence |
| NEKG | Each listed numerical entity | Interpolates outside the entity table |
| DynamicEmbedding | Each allocated key | Allocates a new trainable row |
| FoNE | Shared numerical token and optional projection | Computes fixed decimal-scale Fourier features |
| BitTokens | Shared numerical token and prediction head | Encodes floating-point bits and the reciprocal |
| Prime Fourier Embeddings | Shared neural layers | Computes fixed prime-power features |

The following sections describe the original mechanisms and their output
representations. Entries include transformer models and numerical embedding
or storage methods that could be used around them.

## Exact lookup tables in grokking experiments

Power et al. use a separate symbol for each mathematical object in an equation.
For a finite arithmetic domain, a whole operand or answer occupies one token;
decimal digits and numerical ordering are not supplied. A transformer learns
symbol embeddings and predicts the answer categorically. The paper demonstrates
delayed generalization on several finite operation tables.
[Grokking, 2022, Sections 1 and 2](https://arxiv.org/html/2201.02177v1).

This directly matches our hypothesis on a finite domain. For exact rational
numbers, the lookup key could be the reduced pair $(p,q)$ with $q>0$, so
$1/2$ and $2/4$ share one identity. There is no restriction to small magnitudes:
an explicitly listed large fraction can also have one token. The restriction is
the number of identities in the table.

A newly allocated row has no learned arithmetic meaning merely because its
integer ID is adjacent to existing IDs. A fixed output classifier also needs
that answer in its vocabulary.

## FP15 and a vocabulary of complete numerical values

Charton's FP15 encoding assigns one token to a complete signed numerical value
in bounded scientific notation. The reported configuration uses about
**30,000 vocabulary entries** and rounds coefficients to three significant
digits. For example, $3.14$ is one token, `FP314/-2`.
[Linear algebra with transformers, 2022, Section 2.1 and Table 1](https://arxiv.org/html/2112.01898v2).

This provides an independent learned embedding for each represented value over
a much broader magnitude range than `0..96`. Its output can also be categorical
over complete values. The paper additionally evaluates asymmetric encodings:
FP15 inputs with outputs split into sign, mantissa and exponent tokens.
[Linear algebra with transformers, Sections 4.5 and 4.6](https://arxiv.org/html/2112.01898v2).

FP15 preserves our table requirement for its finite set of values. Rounding
merges other values: arbitrary rational inputs, including $1/3$, cannot all be
recovered exactly. Its numerical accuracy criteria use tolerances; they should
not be read as exact rational arithmetic results.

## Chronos and quantized time series

The original Chronos normalizes a time series, quantizes each scaled observation
into a finite bin vocabulary, and trains a transformer with cross-entropy. The
output is a distribution over bins; decoding maps a predicted bin to its center
and restores the scale.
[Chronos, 2024, Section 3](https://arxiv.org/html/2403.07815v2).

This is a direct example of turning numerical prediction into classification.
The trainable embedding belongs to a **bin**, however. Multiple exact numbers
share it, and a token's value in original units depends on the series scale.
It therefore supports the weaker premise of a discrete numerical vocabulary,
while losing the identity of an arbitrary rational number.

## Prototype embeddings

Jiang et al. construct a finite set of numerical prototypes using a
self-organizing map or a Gaussian mixture model. A number receives a weighted
combination of the prototypes' learned embeddings. In schematic form,

$$
e(x)=\sum_{k=1}^{K}\alpha_k(x)u_k.
$$

The weights depend on the numerical value; the prototype vectors $u_k$ carry
shared learned information. The paper studies numeral representation and
prediction in NLP, including integration with skip-gram, rather than presenting
a transformer for exact rational generation.
[Learning Numeral Embedding, 2020](https://aclanthology.org/2020.findings-emnlp.235/).

This addresses the problem of numbers absent from a vocabulary and gives one
vector per whole input. It replaces an independent vector per number with
shared numerical anchors. An exact output decoder would still need to be
specified for our task.

## LUNA and NumBed

LUNA's NumTok recognizes complete numerical strings and adds or substitutes a
special numerical token. NumBed generates its vector using a character encoder:
CharLSTM uses a bidirectional LSTM; CharFormer combines a transformer encoder,
digit significance information and an LSTM. The main language model receives
one numerical vector, even though its construction reads the number's
characters. The AddBack variant also retains the original text tokens.
[LUNA, 2023 version, Section 3.1](https://arxiv.org/html/2212.02691v2).

This is a close precedent for **processing a number as a whole inside the
main transformer**. Its character encoder shares parameters across numbers;
there is no independent trainable row for every complete value.

The paper focuses on numerical understanding. Its pretraining combines token
classification with regression of numerical sign and log magnitude; it leaves
general number generation for future work. Adapting NumBed to read canonical
`p/q` strings and adding an exact decoder would be a new extension.
[LUNA, Sections 2.1 and 3.2](https://arxiv.org/html/2212.02691v2).

## NumGPT and scientific notation

NumGPT decomposes a number into mantissa and exponent, $x=m10^e$. It combines a
learnable exponent embedding with mantissa features computed from distances to
fixed prototypes. Its implementation has a bounded exponent vocabulary with
overflow and underflow categories.
[NumGPT, 2021, Methods](https://arxiv.org/pdf/2109.03137).

The output separately predicts whether the next item is text or a number.
Numerical generation classifies the exponent and predicts the mantissa through
a continuous head. This is an explicit numerical input/output design, but it
does not assign independent vectors or output classes to all complete numbers.
[NumGPT, Output Layer and Numeral-aware Loss Function](https://arxiv.org/pdf/2109.03137).

Scientific notation helps separate magnitude from precision. A finite-precision
mantissa still cannot preserve every rational value exactly, and mantissa
regression would change our categorical-output premise.

## DICE and numerical geometry

DICE constructs numerical embeddings whose angular relationships encode
numerical distance within a chosen interval. The base construction is
deterministic. The paper also studies initialization with those vectors and an
auxiliary numeracy loss for contextual models such as BERT.
[Methods for Numeracy-Preserving Word Embeddings, 2020, Sections 3 and 5](https://aclanthology.org/2020.emnlp-main.384.pdf).

This suggests a distinction useful for our setup. Using a fixed geometric
function changes the independent-table premise. Initializing separate table
rows geometrically and then training those rows independently can preserve it.
The latter remains a finite vocabulary, and supplies a numerical prior.

For our modular task, ordinary distance and modular arithmetic structure are
different priors. The repository already documents
[geometric initialization and freezing options](embedding-geometry.md);
that functionality does not itself establish which geometry helps rational
arithmetic.

## Learned embeddings of numerical features

Gorishniy et al. embed scalar features before a transformer or MLP mixes them.
Their methods include piecewise linear encodings and periodic features followed
by learned transformations. The piecewise encoding retains position inside an
interval instead of replacing the value with only a bin ID. The periodic
approach uses sine and cosine features, with trainable frequencies in their
experiments.
[On Embeddings for Numerical Features in Tabular Deep Learning, 2022, Section 3](https://proceedings.neurips.cc/paper_files/paper/2022/file/9e9f0ffc3d836836ca96cbf8fe14b105-Paper-Conference.pdf).

These methods produce a vector for each whole scalar, with parameters shared
across that feature's values. Their output is determined by the downstream
classification or regression task. They are useful alternatives to passing
bare floats into an MLP, but they do not satisfy our independent-vector premise.

## xVal and a shared numerical direction

In the basic xVal scheme, every numerical item uses the same `[NUM]` embedding,
scaled by its value: $e(x)=xv_{\mathrm{NUM}}$. A transformer processes those
vectors alongside text. A token head predicts text or a numerical marker, and
a scalar head predicts the associated value using MSE. Later variants broaden
the dynamic range using several numerical directions.
[xVal, Section 2](https://arxiv.org/pdf/2310.02989).

xVal is directly relevant to scientific numerical transformers and accepts new
values through a shared rule. Its basic numerical embedding has one learned
direction, and its numerical output is continuous. Both choices depart from
our current premise; the scheme does not supply exact arbitrary rational
input/output.

## Digit sequences and Abacus embeddings

McLeish et al. keep digit tokens and add learned embeddings for a digit's
position within its number. Their Abacus method uses randomized position
offsets during training and demonstrates generalization to longer addition
problems under the evaluated configurations. Output generation predicts digits
categorically and is evaluated by exact whole-answer matching.
[Transformers Can Do Arithmetic with the Right Embeddings, 2024, Section 3](https://arxiv.org/html/2405.17399v2).

This represents new whole numbers without new vocabulary rows. It does not
satisfy the requirement of an independent vector for each whole number, so it
belongs in the comparison as an alternative premise.

There is also direct rational-arithmetic work: Charton encodes numerator and
denominator integers as digit sequences. In those experiments, comparison
worked well, while simplification and addition in reduced form remained very
poor. This distinguishes successful representation of an exact fraction from
successful learning of the arithmetic and reduction algorithm.
[Learning the greatest common divisor, Appendix A](https://arxiv.org/html/2308.15594v2#A1).

## Precedents that retain a numerical table

### NEKG: learned vectors for numerical entities

NEKG constructs a finite graph of numerical entities connected by neighboring
`isLessThan` relations, then trains TransE embeddings for its nodes. Each
listed value has its own learned entity vector. Numbers outside that list
receive an interpolation of neighboring vectors, rather than a newly trained
independent row. Evaluations cover numerical probes and an NLP application;
this is a numerical embedding method, not a demonstration of rational
arithmetic grokking in a transformer.
[Learning Numeracy: A Simple Yet Effective Number Embedding Approach Using Knowledge Graph, 2021, Algorithm 1](https://aclanthology.org/2021.findings-emnlp.221/).

For our hypothesis, the relevant part is the trained entity table. A possible
extension is to initialize canonical rational rows through a graph, then train
each row freely on arithmetic. This is our proposal: ordering information may
help, but does not supply the division algorithm. Graph pretraining must count
toward the training budget, and must not contain held-out arithmetic answers.

### DynamicEmbedding: growing the table

DynamicEmbedding separates embedding storage from the fixed computation graph.
Its service accepts keys, creates embeddings when keys first appear, and
stores their trainable parameters. It also discusses output retrieval and
negative sampling for a growing collection of output identities.
[DynamicEmbedding: Extending TensorFlow for Colossal-Scale Applications, 2020, Section 3.1](https://arxiv.org/html/2004.08366).

This is an engineering precedent for our strong premise. A canonical fraction
such as `137/991` can be a key with its own row; its components need not fit a
predeclared small integer interval. Storage grows with encountered identities.
The method does not establish numerical semantics, exact arithmetic, or
grokking. Allocating a row and learning how to use it remain separate steps.

Our possible combination is a growing exact dictionary, a shared arithmetic
network, and independently trainable rows. Initializing a new row from a
shared encoder would preserve independent parameters **after allocation**;
using that encoder forever in place of the row would change the premise.

## Recent approaches to one vector per number

These methods are relevant comparisons, but their numerical codes are shared
constructions rather than independent parameters for each complete value.

### FoNE: Fourier features with categorical digit outputs

FoNE uses fixed circular features at decimal scales:

$$
\phi(x,T)=(\cos(2\pi x/T),\sin(2\pi x/T)),\qquad
\operatorname{FoNE}(x)=[\phi(x,10^i)]_{i=-n+1}^{m}.
$$

Here $m,n$ are global limits on integer and fractional digits. Features are
padded or projected and added to a shared `[NUM]` embedding. The output uses
parallel ten-class predictions for digit positions, trained with
cross-entropy. It avoids a class for every possible complete answer.

The revised paper reports 99% exact-match accuracy on six-digit decimal
addition with 6,400 training examples and a 37.55-million-parameter model.
This is evidence for that bounded arithmetic task, not arbitrary rational
division or delayed generalization. The configured decimal precision cannot
represent every rational exactly.
[FoNE, version 2, April 2026, Sections 3 and 4](https://arxiv.org/html/2502.09741v2).

For us, the useful design idea is a structured vector plus discrete output
decisions. Using these features only to initialize subsequently free table
rows would be a different experiment from published FoNE.

### BitTokens: binary floating-point codes, including the reciprocal

BitTokens encodes the 64 bits of an IEEE 754 float64 value as signed binary
features. It also concatenates the encoding of **the precomputed reciprocal
$1/x$**, producing 128 features before padding and adding a shared `[NUM]`
embedding. Its numerical output head predicts bits with sigmoid activations
and binary cross-entropy, then thresholds and decodes them.

The paper studies arithmetic with a curriculum and large synthetic datasets.
Its significant-digit score and exact-match score are distinct metrics.
Exact recovery of a float64 bit pattern still does not recover arbitrary
rationals: for example, float64 stores a rounded approximation to $1/3$.
[BitTokens, version 2, May 2026, Sections 2, 5 and 6](https://arxiv.org/html/2510.06824v2).

For comparison with our division experiments, the reciprocal is a material
extra input. An ablation must remove it before attributing success to the
network learning division from the two original values. There is no
independent row per number, and this result is not evidence of grokking on
an expanding rational vocabulary.

### Prime Fourier Embeddings: modular features with a narrower claim

PFE uses fixed sine/cosine features with prime-power periods. The reported
models use a shared MLP per prime and a categorical classifier, rather than
a transformer or an independent table of numerical values. Experiments cover
modular addition, with a single seed and splits over distinct integer input
pairs. The theoretical block structure assumes an equivariant linear map;
the paper explicitly leaves convergence to such a solution open.
[Prime Fourier Embeddings, version 3, October 2026, Sections 3 and 4](https://arxiv.org/html/2606.23044v3).

Our interpretation requires two qualifications. References to harmonic
analysis on $\mathbb{Q}$ do not establish an exact rational input/output
experiment. Also, distinct integer pairs can reduce to the same pair modulo
the task modulus. An operation-table generalization claim therefore needs a
split audited at the level of **residue pairs**, as well as original pairs.
That is a proposed additional check, not a claim that the reported data
contain duplicate original examples.

## What the grokking papers establish

### GrokTransfer: transfer learned embeddings within the same vocabulary

GrokTransfer first trains a weaker model, then transfers its numerical
embeddings into a larger model on the same task and vocabulary. The published
construction is

$$
E_T=A B,\qquad A\leftarrow E_W,
$$

with both factors trainable. Each identity retains a separately trainable row
in $A$, but $B$ shares a projection and constrains the resulting table's rank.
The experiments include transfer between different model architectures.
The paper also observes that informative Fourier features can remove the
delay between fitting and generalization on modular addition.
[Let Me Grok for You, ICLR 2025, Sections 2.1 and 2.2](https://arxiv.org/html/2504.13292).

This supports investigating learned embeddings and their initialization. It
does not demonstrate transfer to new number identities or to all of
$\mathbb{Q}$. Copying a projected table into a fully independent target table
would be our ablation, not the published factorization. Fast generalization
would be a useful outcome even if the conspicuous grokking delay disappeared.

### Embedding learning rates and coverage

A separate study examines embedding update frequencies and their interaction
with the rest of a modular-arithmetic model. Its experiments include modular
division and use an embedding learning rate ten times the remaining network's
rate. Its discussion of one-hot inputs still includes a learned first linear
map: a column of that map acts as a learned embedding.
[Mechanistic Insights into Grokking from the Embedding Layer, 2025, Sections 3 and 5](https://arxiv.org/html/2505.15624).

This motivates tuning the embedding learning rate separately and measuring
coverage of every identity. The factor ten is an experimental setting, not
a universal prescription. The one-hot setup also cannot be read as evidence
about our previous model receiving two ordinary floating-point scalars.

### Repetition and exact reduction

Charton and Kempe show that reusing finite datasets can improve mathematical
learning relative to continually supplying new examples at a comparable
training budget. Their GCD evaluation balances answer values from 1 to 100:
accuracy on naturally sampled integer pairs can conceal weaknesses because
small GCD values dominate.
[Emergent properties with repeated examples, Sections 3 and 4](https://arxiv.org/html/2410.07041).

For rational arithmetic, this suggests tracking accuracy by denominator and
reduction difficulty, and comparing repeated versus fresh training examples
explicitly. It does not establish that independent embeddings are necessary,
or that the fraction-reduction problem has been solved generally.

## Consequences for exact rational input and output

For our strong hypothesis, an exact rational table should identify each value
by a canonical pair $(p,q)$, with $q>0$, $\gcd(|p|,q)=1$, and zero represented
as $(0,1)$. The model predicts a class whose associated pair is the answer.
No floating-point rounding is needed to recover the label.

Even restricting values to $[0,1]$ leaves infinitely many rationals. A finite
table must restrict the number of identities, for example through bounds on
numerators and denominators or an explicit list. Precision limits such as
FP15 instead introduce approximation.

The vocabulary can grow as numbers are encountered. This preserves independent
parameters for every allocated value, but requires more storage and training
for the new rows. At every finite stage it still covers a finite subset of
$\mathbb{Q}$. A collision-resolving hash map keyed by the full canonical
fraction can preserve identities exactly. Assigning values to a fixed set of
embedding buckets without resolving collisions merges identities. Generating
new vectors through a shared encoder changes the premise.

For scale, an input table with $V$ rows of width $d$ needs $4Vd$ bytes in
float32. With float32 gradients and two Adam moment arrays, those four arrays
alone need $16Vd$ bytes. These are calculated storage costs, not measurements:

| Number of values at width 128 | Embedding weights | Weights, gradients and two moments |
| ---: | ---: | ---: |
| 10,000 | 5.12 MB | 20.48 MB |
| 1,000,000 | 512 MB | 2.048 GB |
| 10,000,000 | 5.12 GB | 20.48 GB |

Units are decimal. These figures exclude the model body, activations, logits
and any separately parameterized output table. Tying input and output weights
can share the table; predicting over all $V$ classes still has a cost.

Input and output coverage must be designed together. For ordinary division,
two input fractions with component magnitudes bounded by $N$ can produce an
answer whose components reach $N^2$ before reduction. A finite input vocabulary
need not contain all answers. This differs from our division modulo 97, where
valid division stays inside the same finite field.

## The output vocabulary is a separate scaling problem

Making input lookup sparse does not make a full output softmax sparse.
For a batch of $B$ examples, explicitly materialized float32 logits take
$4BV$ bytes. With $B=512$ and $V=10^6$, that is **2.048 GB for logits alone**.
A dense linear head takes work proportional to $BVd$; its training gradient
also reaches output rows for classes absent from the targets. Tying that head
to the input table shares weights, but does not remove this dense computation.
Chunking can reduce peak logit storage without reducing the number of scores.

Adaptive softmax is a published way to organize output classes by frequency
and reduce computation, with reported gains on language modeling.
[Efficient softmax approximation for GPUs, 2017](https://proceedings.mlr.press/v70/grave17a.html).
Its advantage should be measured on our arithmetic label distribution;
approximately uniform residues differ from a language vocabulary. Structured
classification still selects from registered complete values.

Numeral retrieval is another precedent. Sakamoto and Aizawa use nearest
neighbor language models to retrieve numeral candidates from contextual
examples, including numerals outside the language model's vocabulary. The
paper explicitly discusses failure when the correct numeral is absent from
the datastore.
[Predicting Numerals in Text Using Nearest Neighbor Language Models, 2023, Section 6.3](https://aclanthology.org/2023.findings-acl.295/).

Thus an external candidate store can enlarge the available answer set, but
retrieval alone cannot create an answer missing from that store. For an
arithmetic application, candidate coverage and retrieval recall must be
reported separately from the network's ranking accuracy. Sampling negatives
also changes training; it is not automatically equivalent to full softmax.

## A conditional route from finite vocabularies to exact fractions

This is **our proposed architecture**, based on standard modular arithmetic
and rational reconstruction. It is not a result demonstrated by the embedding
papers above, and it **relaxes the independent-vector-per-fraction premise**:
the independently learned symbols would be residues in several finite fields.

For a reduced fraction $a/b$ and a prime $\ell$ not dividing $b$, define

$$
\rho_\ell(a/b)=a\,b^{-1}\pmod\ell.
$$

Each channel can use the same kind of finite categorical task as our modulo-97
experiments. Given fractions $x,y$, a division model predicts
$\rho_\ell(x)/\rho_\ell(y)$ modulo $\ell$. Combining predictions for distinct
primes with the Chinese remainder theorem yields one residue $r$ modulo
$M=\prod\ell$.

To recover the fraction, specify bounds $|a|\le A$ and $1\le b\le B$ in
advance. Provided $\gcd(b,M)=1$, a sufficient uniqueness condition is

$$
M>2AB.
$$

The reason is elementary: two candidate fractions with the same residue would
satisfy $M\mid(a_1b_2-a_2b_1)$, while the absolute value of that difference is
at most $2AB<M$. It must therefore be zero, so the fractions are equal.
Existence is a separate condition: incorrect predicted residues can have no
fraction within the bounds, or reconstruct to the wrong one.

Rational reconstruction is an established algorithm, with an implementation
and the symmetric bound in the
[SageMath reference](https://doc.sagemath.org/html/en/reference/rings_standard/sage/arith/misc.html#sage.arith.misc.rational_reconstruction).
It supplies an exact symbolic decoder; the neural network would learn the
finite-field operation, not the reconstruction algorithm.

### Worked division example

Suppose all input numerators have magnitude at most 8 and all denominators
are at most 8. A nonzero-divisor quotient has reduced component magnitudes at
most 64. Choose primes 97 and 101: all nonzero input components are invertible,
and $M=9797>2\cdot64^2=8192$ is sufficient for reconstruction.

Consider $3/5\div7/8=24/35$:

| Channel | Encoding of $3/5$ | Encoding of $7/8$ | Correct predicted quotient |
| --- | ---: | ---: | ---: |
| Modulo 97 | 20 | 13 | 9 |
| Modulo 101 | 41 | 64 | 18 |

The two predicted residues combine to $r=2240$ modulo 9797. The relation
$35\cdot2240\equiv24\pmod{9797}$ reconstructs the exact answer $24/35$.
Locally checked with integer arithmetic: all **5,039** reduced fractions with
$|a|\le64$ and $1\le b\le64$ have distinct encodings under these two moduli,
including zero as $0/1$. This is an encoding check, not a trained-model result.

The two channels need 97 and 101 residue identities respectively, rather than
an independent identity for each of those 5,039 fractions. Their combinations
carry the information; parameters are shared between fractions. That sharing
is precisely why this proposal departs from our strong hypothesis.

### Conditions that the experiment would need to enforce

- **Valid primes:** input denominators and the divisor's numerator must be
  nonzero modulo each selected prime. Selection or masking must use inputs,
  not the correct answer. A fixed prime collection cannot handle every
  rational input this way.
- **Declared bounds:** choose the reconstruction modulus using input-domain
  bounds, before seeing the answer. Larger domains require a larger product
  of primes and potentially additional learned channels.
- **Exact end-to-end evaluation:** measure exact canonical fractions, failed
  reconstructions, and individual residue errors. One erroneous channel can
  spoil the complete answer. Extra channels could provide checks, but do not
  automatically correct errors.
- **Arithmetic outside the network:** modular input conversion uses exact
  inversion of input denominators; the output uses CRT and reconstruction.
  These are substantial symbolic computations and must be included in the
  system description. Compare against a fully symbolic rational baseline.
- **Meaningful held-out examples:** unseen fractions can still map to residue
  pairs seen during training. Report coverage and accuracy on genuinely
  withheld operation-table cells in each channel as a separate evaluation.

This offers a finite representation for every declared bounded problem.
Handling all of $\mathbb{Q}$ would require resources to grow with the requested
domain or precision; a fixed collection of finite vocabularies is insufficient.

## Proposed experiments, in priority order

These are proposals, not additional runs. The main path retains independent
rows; the modular reconstruction route is conditional on relaxing that choice.

### 1. Isolate the embedding contribution on division modulo 97

Start from a known successful categorical model and its established split.
Keep the network body, output classes and cross-entropy objective fixed.
Compare learned random table rows, frozen random rows, and fixed numerical
features projected to the same input width. Include ordinary scalar inputs
with this categorical output as a control: the previous scalar MLP changed
the output and loss as well as the input representation.

Report parameter counts and equal tuning budgets, not just the best run.
Use at least three seeds with matched data splits. This separates the value
of training independent rows from benefits of a discrete output or a useful
fixed code, while retaining the accepted table-based model as the baseline.

### 2. Improve the independently trainable table

Compare random initialization with learned small-model initialization and a
numerical graph initialization. Keep rows fully trainable after initialization;
test the published low-rank transfer as a separate variant. Count pretraining
cost and use a separate validation set to select checkpoints.

Tune the embedding learning rate independently from the network's rate.
Record per-identity example counts, gradient norms and embedding norms, so a
rarely updated value is not mistaken for an architectural failure. Keep
weight decay as a measured hyperparameter rather than assuming it is required.

### 3. Build the exact bounded-rational control

Define a finite domain of canonical fractions and predeclare an output domain
large enough for the operation. Use arbitrary-precision integers for keys,
data generation and answer verification; the learned vectors can remain
float32. Class identity, not the vector's floating-point precision, preserves
the exact output fraction.

Separate these evaluation settings:

| Setting | What is new? | What it measures |
| --- | --- | --- |
| Known identities, unseen equations | Operand combinations | Learning the operation within a learned vocabulary |
| Added identities before adaptation | Newly allocated rows | Initialization quality; random rows have no supplied arithmetic meaning |
| Added identities after adaptation | A fixed budget of training equations | Cost of learning new rows and retention on old identities |
| Larger component bounds | Magnitude or denominator complexity | Extrapolation together with vocabulary and resource growth |

In the first setting, require positive training examples for every evaluated
output identity and coverage of input identities in each operand role. Report
any excluded or uncovered cases separately. A classifier failing on an answer
class never positively trained is a different issue from failing on an unseen
equation composed of known values.

For vocabulary growth, compare random new rows with shared-encoder
initialization followed by free row training. Predeclare adaptation budgets,
for example 1, 4, 16 and 64 equations per new identity, and also measure old
identity accuracy. No automatic arithmetic ability should be attributed to
the act of inserting a key into the dictionary.

### 4. Evaluate the optional modular reconstruction route

Only after choosing that weaker representation, test the bounded example
above with learned categorical division in both fields. A modulo-97 result
does not establish a modulo-101 result. First verify each operation-table
split and the exact decoder independently; then measure complete fraction
accuracy. Keep this result distinct from learning fraction simplification
inside a network.

Across these studies, retain train and held-out learning curves, exact answer
accuracy, data and compute budgets, and variation across seeds. Delayed
generalization would document grokking; immediate reliable generalization
would also be useful. None of the representation schemes by itself guarantees
either outcome.
