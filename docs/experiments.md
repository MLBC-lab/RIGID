# Optional training and baseline experiments

The `rigid.experiments` module supplies complete training and inference workflows
for graph proposals, a conditional fingerprint VAE, and observed-profile nearest
neighbors. Each workflow records its configuration, input provenance, and evaluation
population. See [study materials](SCIENTIFIC_CONTRACT.md) for manuscript-specific inputs.

All workflows consume the same prepared dataset described in the data guide.
Training requires nonempty `train` and `validation` partitions and rejects
compound-identity leakage before fitting. Set `scaffold_holdout=True` on a training
function or the nearest-neighbor function to require the stricter scaffold audit.
The actual audit is recorded in training manifests. A compound-only audit must
not be reported as scaffold-held-out evaluation.

## Graph generator

```python
from rigid.config import TrainConfig
from rigid.experiments import train_generator

config = TrainConfig(epochs=30, hidden_dim=128, batch_size=16, seed=17)
manifest = train_generator(
    "data/prepared",
    "runs/generator",
    config,
    max_atoms=40,
    conditional=True,
    scaffold_holdout=True,
)
```

The generator uses the graph construction implementation described in
[Models](models.md). Training and validation compounds are checked against its
explicit vocabulary and graph-action limits. Unsupported examples are recorded
in `exclusions.json` with compound IDs, profile IDs, and reason counts for each
partition. Filtering runs on the dataset's standardized parent structure;
earlier documented salt removal or neutralization is part of data preparation.
The graph-trace filter does not inspect test compounds. If no usable training or
validation examples remain, training stops with the exclusion report preserved.

Conditional training uses one example for each retained profile. Its condition
is the normalized expression vector paired with that profile's compound.
Normalization is fitted using only retained training examples. Validation
conditions use the saved training transform. The optional unconditional mode
uses one example per retained compound, preventing replicate counts from
overweighting a structure. Both target-conditioned generation and unconditioned
proposal followed by target-specific ranking are supported experimental choices.

Teacher forcing optimizes mean action cross-entropy for each example, averaged
across examples. AdamW, gradient clipping, deterministic settings, batch size,
learning rate, and early-stopping patience come from `TrainConfig`. The checkpoint
with the lowest validation loss is retained. Neither the training loss nor the
validation-selection loss uses test profiles.

The output directory must be empty. A completed run contains:

| Artifact | Content |
| --- | --- |
| `model.pt` | Architecture configuration, selected weights, transform, training identities |
| `model.pt.sha256.json` | Checksum verified before loading |
| `manifest.json` | Input hashes, software versions, splits, exclusions, objective |
| `history.json` | Training and validation losses by epoch |
| `exclusions.json` | Unsupported chemistry and retained-example counts |

### Sampling from a saved generator

```python
from rigid.data import load_dataset
from rigid.experiments import generate

dataset = load_dataset("data/prepared")
index = next(i for i, profile in enumerate(dataset.profiles) if profile.split == "test")
report = generate(
    "runs/generator/model.pt",
    n=128,
    seed=42,
    target=dataset.expression[index],
    genes=dataset.genes,
)
```

`generate` verifies the checkpoint checksum, loads tensor weights on CPU, checks
the gene manifest, applies the saved transform, and samples with a local seed.
The target must be one raw aligned expression vector. An unconditional checkpoint
requires neither a target nor a gene list and rejects supplied target arguments.

The returned dictionary includes every attempt, validity, uniqueness, failures,
and the number of unique structures absent from the training set. That last count
uses exact canonical SMILES; it is **not scaffold novelty**. Validity is divided
by attempts, while uniqueness is divided by valid attempts. Nontermination is
counted as a failed proposal. Compatibility scoring, stress-aware reranking,
medicinal-chemistry filters, and uncertainty assessment remain downstream steps.

## Conditional fingerprint-VAE baseline

```python
from rigid.experiments import train_vae, screen_vae

train_vae("data/prepared", "runs/vae", config, scaffold_holdout=True)
report = screen_vae("runs/vae/model.pt", "data/prepared", split="test", samples=32)
```

Defaults are radius-2 Morgan fingerprints with 2,048 bits, latent dimension
`max(4, hidden_dim // 2)`, and KL weight `beta=0.01`. These choices are saved in
the manifest and can be changed through the Python API or the `train-vae` flags
`--fingerprint-bits`, `--latent-dim`, and `--beta`. Report them with the baseline
results so that the exact model and sampling procedure can be rerun.

Training minimizes summed Bernoulli fingerprint reconstruction loss plus beta
times KL divergence, averaged across profiles. Validation posterior sampling
uses a fixed seed for each validation batch across epochs, so random resampling
does not change the checkpoint-selection criterion. The training RNG state is
restored after each validation calculation. Fingerprints are prepared only for
training and validation compounds during fitting; gene normalization uses only
training profiles.

At screening time, the model samples fingerprint distributions from the prior
conditioned on each requested target. It ranks the provided molecular catalog
using the mean soft-Tanimoto similarity to these samples. The catalog contributes
chemical structures, not candidate-response profiles. Candidate scoring uses
chunks of 1,024 fingerprints and stable compound-ID tie breaking. Checkpoint
training structures are checked against target source structures to reject
overlapping held-out queries. Gene order and namespace must match.

This is a library-constrained baseline. A probabilistic fingerprint does not
uniquely specify a molecular graph, and this VAE does not generate SMILES. The
training checkpoint contains the fingerprint schema, model configuration,
normalizer, training identities, and selected weights. Screening returns rankings,
retrieval metrics, checkpoint hash, source provenance, and the requested sample
count. The top-k cutoff comes from the saved training configuration.

## Connect proposal generation to RIGID ranking

The executable example connects both trained checkpoints for one target:

```sh
python examples/design_candidates.py \
  --dataset data/prepared \
  --ranker runs/ranker/model.pt \
  --generator runs/generator/model.pt \
  --target-id TARGET_PROFILE_ID \
  --output runs/design.json \
  --count 128 --top-k 50 --seed 17
```

Replace `TARGET_PROFILE_ID` with a profile ID present in the dataset. The script
verifies both checkpoint checksums and gene manifests, then passes the target
expression and experimental context through generation and ranking. The hybrid
pool combines the existing molecular library with unique, valid generated
structures. Canonical duplicates are merged before ranking, and each selected
candidate records whether it came from the library, generation, or both. This
example does not perform analog optimization.

The ranker receives no target source-compound or MOA labels. The full generation
attempt report is retained alongside pool accounting, scores, descriptors,
checkpoint hashes, input hashes, and the training-source labels of both models.
If the input or either model was trained on synthetic data, the complete output
is explicitly labeled a synthetic demonstration. An existing output file is
never overwritten.

The example reports chemistry descriptors and alerts but applies no hard
medicinal-chemistry cutoff. It accepts any named target, including training
profiles; it is a design workflow, not a held-out performance benchmark. Use
the dedicated evaluation APIs and explicit partition checks for performance
claims.

## Observed-response nearest neighbors

```python
from rigid.experiments import nearest_neighbor

report = nearest_neighbor("data/prepared", split="test", k=50)
```

This baseline compares each target with observed **training** expression profiles.
It fits gene-wise normalization on training data, computes Pearson correlation
across genes, and assigns each training compound its maximum correlation over
that compound's reference profiles. Constant reference profiles are excluded
because their correlation is undefined. A constant target produces an explicit
error. No validation or test candidate-response profiles enter the reference
catalog.

The output records the exact reference profile IDs and any excluded constants.
The default does not match cell, dose, or time; differences in those contexts can
confound retrieval and should be considered when interpreting this baseline.

The default candidate catalog contains only training compounds, making exact
recovery of a strictly compound-held-out target impossible. For a comparison
using a separately measured reference catalog, provide that dataset explicitly:

Save a reference policy, for example as `reference-policy.json`:

```json
{
  "reference_splits": ["train"],
  "match_cell": true,
  "dose_tolerance_um": 0.0,
  "time_tolerance_h": 0.0,
  "aggregation": "max",
  "chunk_profiles": 1024
}
```

```sh
rigid neighbors data/prepared --reference-dataset data/reference --reference-config reference-policy.json --top-k 50 --output runs/neighbors.json
```

This mode can retrieve held-out compound identities through independent response
measurements. It checks profile-ID overlap, exact copied expression vectors, gene
order, and identity conflicts. Cell matching, dose/time tolerances, and per-compound
`max` or `mean` aggregation are configurable. Declare the candidate universe and
available measurements before comparing methods: access to measured candidate
responses differs from structure-only prediction. See the
[reference-catalog guide](neighbors.md) for the full protocol and provenance fields.

## Verification and provenance

`tests/test_experiments.py` performs small real training runs on synthetic data,
then reloads and exercises the saved checkpoints. It tests chemistry exclusion
records, conditional gene-order enforcement, deterministic sampling, VAE
screening, and checksum rejection. A leakage test changes every test expression
value and confirms that the fitted generator parameters remain identical. Another
test changes validation responses and confirms that the observed-training
nearest-neighbor rankings do not change.

Synthetic source provenance is carried into run manifests and reports. Software
checks and biological validation are separate: `reproduces_manuscript` remains
false in this release. See [research scope and study materials](SCIENTIFIC_CONTRACT.md).
