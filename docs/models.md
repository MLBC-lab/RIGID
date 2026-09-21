# Model architectures and configuration

RIGID separates candidate ranking from molecular proposal and forward validation.
This guide specifies the implemented layer structure, input features, objectives,
and defaults. Save these settings with each experiment; the defaults are not a
verified study configuration. Study materials are described in the
[research scope](SCIENTIFIC_CONTRACT.md).

## Dual-encoder ranking

`rigid.models.RIGIDRanker` combines a directed molecular graph encoder, an
expression-profile encoder, and an MLP ranking head.

| Component | Input | Output | Default |
| --- | --- | --- | --- |
| D-MPNN | Atom features, directed bonds, reverse-bond map | Molecular embedding | 128 dimensions, depth 3 |
| Target encoder | Ordered expression vector | Target embedding | 978 → 256 → 128 |
| Ranking head | Both embeddings and optional auxiliary features | Compatibility score | 256 → 128 → 1 hidden/output widths |

The profile is one vector per target, after gene-order alignment and normalization
fitted on training data. A batch of profiles is a matrix; it is not a separate
spatial or image input. A dimension other than 978 is supported when specified
explicitly in the configuration and data manifest.

### Directed message passing

Each chemical bond has two directed edges. An edge from atom `v` to atom `w` is
initialized from the source atom features and the bond features. At every update,
its incoming message is the sum of messages entering `v`, excluding the reverse
edge `w → v`. This prevents an immediate echo along the same bond. Updates use a
shared learned projection with a skip from the initial bond state. Incoming bond
states are then combined with atom features and mean-pooled by molecule.

This is the Chemprop-style bond-message construction described in the
[Chemprop message-passing documentation](https://chemprop.readthedocs.io/en/main/autoapi/chemprop/nn/message_passing/index.html).
The implementation is independent; feature vectors and checkpoint keys are not
compatible with upstream Chemprop checkpoints. Input graph validation checks
reverse-edge involution, bond endpoints, finite features, and graph membership.
Single-atom molecules are supported. Empty molecules are rejected.

### Profile and ranking networks

The target encoder uses linear layers, GELU, dropout, and output layer
normalization. The head receives:

- The molecular and target embeddings.
- Their elementwise product and absolute difference.
- Their cosine similarity.
- An explicitly configured vector of similarity, program, stress, or other
  auxiliary features, when enabled.

These interaction features are part of the ranking-head input contract.
Auxiliary features must have a fixed, documented order and must be available for
new candidates at inference.
Do not supply an observed held-out compound response as a predictor feature when
the evaluation is intended to represent unseen-compound prediction.

The output is an unbounded ranking score. It is not a calibrated probability of
causing a response, a binding affinity, or a measure of therapeutic efficacy.
External calibration must use a dedicated, disjoint calibration partition.

```python
import torch

from rigid.chem.graph import (
    ATOM_FEATURE_DIM,
    BOND_FEATURE_DIM,
    graph_batch_from_smiles,
)
from rigid.models import RIGIDRanker

model = RIGIDRanker(ATOM_FEATURE_DIM, BOND_FEATURE_DIM, profile_dim=978)
graph = graph_batch_from_smiles(["CCO", "CC(=O)N"])
target = torch.zeros(1, 978)  # Replace with an aligned, normalized target.
model.eval()
with torch.inference_mode():
    scores = model(graph, target)
```

For repeated retrieval, cache `model.molecule_encoder(graph)` and use
`model.score_library(molecules, targets, chunk_size=1024)`. The result has shape
`[number_of_targets, number_of_candidates]`. The head processes bounded chunks,
although the returned score matrix itself still occupies that full size. Callers
with very large catalogs should stream target groups and retain only their top-k
results. Cache provenance must include the encoder checkpoint, feature schema,
and canonical candidate identities.

### Ranking losses

`pairwise_rank_loss` penalizes a negative candidate scoring above its paired
positive. It supports nonnegative example weights and a score margin.
`listwise_rank_loss` minimizes cross-entropy against normalized graded relevance
within a target's candidate set. This permits exact and same-MOA positives to
receive different declared relevance values. A target with no positive label is
rejected rather than silently contributing no learning signal.

Neither objective chooses negative compounds. Matched examples, related
mechanisms, scaffold decoys, and mismatches must be assembled by the training
pipeline without drawing evaluation examples into training. Loss gradients,
small-problem fitting, batch isolation, and atom/bond permutation invariance are
covered in `tests/test_models.py`.

## Forward-response controls

Two independent trainable implementations are provided:

1. `NeuralProfilePredictor`: D-MPNN molecular embedding plus optional context,
   followed by an MLP that predicts an expression vector.
2. `GraphTransformerPredictor`: atom-centered graph messages followed by masked
   Transformer attention over the atoms and a profile-output layer. It has its
   own parameters and no shared ranking encoder. Atom positions are not encoded,
   so changing atom enumeration cannot change the prediction.

Both accept a configured `context_dim` and a context matrix containing externally
prepared cell, dose, and time features. Context encoders and transforms must be
fitted on the training partition. The graph/Transformer implementation uses dense
attention with quadratic cost in the largest molecule in the batch.

`NeuralProfilePredictor` is a graph regressor, not the ExPO neural operator.
ExPO code and checkpoints are not bundled; that named control requires a separately
validated ExPO integration. The two supplied predictors have independent parameters
but use the same training data. Independent empirical measurements remain necessary
to assess agreement with observed responses.

`masked_profile_loss` computes mean squared error only at observed target genes.
NaN targets and an optional mask are supported. If no observed genes remain, the
loss raises an error. Missing genes must still be imputed before the inverse
target encoder, which requires a complete input vector.

## Conditional-VAE baseline

`ConditionalFingerprintVAE` learns a posterior over a molecular latent variable
from a fingerprint and a desired profile. The decoder predicts Bernoulli
fingerprint probabilities given a sampled latent variable and the target
embedding. Its objective is summed fingerprint binary cross-entropy plus a
weighted KL divergence to a standard normal prior.

`rank_library` compares sampled fingerprint probabilities with actual library
fingerprints using mean soft-Tanimoto similarity. The method returns library
scores; it cannot decode a fingerprint into a unique molecule or generate valid
SMILES. This makes it an explicit **library-constrained conditional-VAE baseline**.
Record decoder type, latent size, beta, fingerprint schema, and sampling settings
when reporting this baseline; they define the comparison, not merely its runtime.

## Optional autoregressive graph proposal

`rigid.generation.AutoregressiveGraphGenerator` implements a trainable graph
proposal policy inspired by GraphINVENT's add/connect/terminate action structure.
See the [GraphINVENT repository](https://github.com/MolecularAI/GraphINVENT) for
the upstream software and references. The policy here has its own feature and
action definitions and is not checkpoint-compatible with upstream GraphINVENT.

At every step, an atom-embedding/message-passing network encodes the partial
graph. A pooled graph embedding drives a stop head; atom/graph features drive an
add head; symmetric atom-pair/graph features drive a ring-connection head. A single
softmax normalizes logits across all currently legal actions. Thus adding an
atom, closing a ring, and terminating compete in the same probability space.

The model is trained with teacher forcing on canonical breadth-first construction
traces, including ring closures and a final stop action. `teacher_forcing_loss`
returns mean action cross-entropy and supports fine-tuning the same model after a
checkpoint is loaded. No pretrained weights are included. A newly initialized
generator produces untrained proposals; test-set validity alone does not imply
learned chemical quality.

The default proposal model is unconditional. RIGID target specificity is supplied
by the subsequent dual-encoder scorer and reranking filters, matching the
manuscript's propose-score-rerank description. Setting `condition_dim` enables a
target-conditioned extension, which must be trained with aligned conditions and
reported as an explicit experimental choice.

### Chemical domain and action masks

The implementation supports connected neutral heavy-atom graphs using
`C, N, O, F, P, S, Cl, Br, I` and single, double, or triple bonds. Aromatic training
molecules are represented with Kekule bond orders and re-sanitized by RDKit.
Valence masks prevent illegal additions, duplicate bonds, self-bonds, and
exceeding the configured maximum graph size. Final RDKit sanitization remains a
separate check.

The current neutral-valence vocabulary deliberately has no formal-charge,
stereochemical, isotopic, radical, salt, or hypervalent representation. Training
examples with these features are rejected explicitly; they must be audited and
reported as excluded examples. The model must not be described as covering the
entire CMap chemical space. Extending the action vocabulary requires changing the
trace encoder, masks, network output heads, checkpoints, and tests together.

Explicit valence masks enforce this implementation's supported chemical domain;
they are not an exact upstream GraphINVENT rule.
Action enumeration and teacher forcing are implemented for clarity and molecular
scale workloads. Large generator training runs should profile these operations
before investing in distributed training.

```python
from rigid.generation import AutoregressiveGraphGenerator, GeneratorConfig

generator = AutoregressiveGraphGenerator(GeneratorConfig(hidden_dim=128))
loss = generator.teacher_forcing_loss(["CCO", "CC(=O)N", "c1ccccc1O"])
loss.backward()
proposals = generator.sample(16, seed=42)
```

Sampling restores the model's previous train/eval mode and accepts a local random
seed. A graph that reaches `max_steps` without selecting stop is recorded as a
failure, not silently accepted. Each proposal records termination, validity,
step count, log probability, and a rejection reason when applicable.

`select_candidates` deduplicates canonical proposals, applies a caller-supplied
chemistry policy, and retains the highest scoring molecules. Its report separates
attempts, valid graphs, unique valid molecules, and accepted molecules. Validity
uses all attempts as its denominator; uniqueness uses all valid proposals. Filter
or scorer failures propagate, preventing infrastructure errors from being
misreported as chemical invalidity. QED, synthetic accessibility, alerts,
scaffold novelty, stress burden, and uncertainty belong in the external selection
policy and must be documented for each experiment.

## Checkpoints, precision, and claims

Top-level models expose a JSON-compatible `config` dictionary. Save it with the
state dictionary, feature/gene schema, training manifest, seed, and software
versions. Generator configuration is restored with
`GeneratorConfig(**saved_config)`. Other top-level model configurations can be
passed directly as constructor keyword arguments. Cross-architecture state
dictionaries are not interchangeable.

Inputs use float32 on the model's device; CPU is the default. GPU execution is
available through normal PyTorch device placement. A seed supports repeatable
sampling within a recorded environment, not universal bitwise equivalence across
hardware or PyTorch releases. Pretrained weights are not bundled. Record the split,
configuration, selected checkpoint, and provenance for each reported evaluation.
