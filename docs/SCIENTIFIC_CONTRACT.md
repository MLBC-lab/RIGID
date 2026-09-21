# Research scope and study materials

## Source and scope

This toolkit implements the computational workflow described in the manuscript
*RIGID: Robust Inverse Gene-Expression Design for Stress-Aware Transcriptomic Candidate
Prioritization*, by Austin Spadaro, Iman Dehzangi, and Alok Sharma.

The source release includes model implementations, data utilities, evaluation methods,
and synthetic software tests. It does not bundle or independently verify the manuscript's
experimental results. Reproducing an endpoint requires the study inputs, fitted models,
and evaluation protocol in addition to executable code.

## Mapping from manuscript to software

| Manuscript method | Implementation | Important boundary |
| --- | --- | --- |
| 978 landmark-gene target signature | `data.schema.GeneManifest`, `models.encoders.TargetEncoder` | Experimental runs must supply the correct IDs and ordering; smoke tests use 32 synthetic genes |
| D-MPNN molecular encoder and target/ranking MLPs | `models.encoders`, `models.ranker` | Widths, pooling, head interactions and optimizer are recorded implementation settings |
| Matched, mechanism-related, scaffold-related and mismatched ranking examples | `training.train`, `models.losses` | Candidate sampling and graded relevance must be fixed in the evaluation protocol |
| GraphINVENT-based proposal stage | `generation`, `experiments.train_generator` | GraphINVENT-inspired action policy; incompatible with upstream checkpoints |
| Independent forward operators including ExPO | `models.forward` | D-MPNN regressor and graph/Transformer control are supplied; ExPO is not bundled |
| Library, hybrid and de novo candidate pools | `pipeline.RIGIDPipeline.rank`, generation sampling/selection utilities | Record the source and composition of each candidate pool |
| Conditional VAE comparison | `models.vae`, `experiments` | Fingerprint-generating/library-retrieval baseline, not a SMILES decoder |
| Stress removal, program emphasis, interpolation, disease reversal | `data.targets` | Gene sets and disease cohort contrasts are supplied by investigators |
| Retrieval, scaffold, stress and expression metrics | `evaluation.metrics` | Missing labels have explicit denominators; no invented annotations |
| Bootstrap, Jaccard, FDR and calibration | `evaluation.statistics`, `calibration`, `controls` | Caller must supply valid independent units, calibration partitions and cohort provenance |
| Designability | `evaluation.ranking.DesignabilityModel` | Explicit weights and validation-fitted min-max normalization |

## Study materials and reproducibility

Contact the manuscript's corresponding authors or the repository maintainer to request
the materials below. They are not bundled in this source release. Availability, permitted
use, and redistribution rights must be confirmed with the provider; the repository does
not guarantee that every requested artifact can be supplied.

1. Exact versions and accessions of CMap/LINCS compound-response profiles, exclusions,
   cell/dose/time processing, the 978-gene manifest, and data-use permissions.
2. Exact compound identities, standardization rules, mechanism labels, aliases and split
   manifests used for compound, scaffold and external holdouts.
3. Disease cohort accessions and preprocessing for psoriasis, ER-positive breast
   cancer and lung adenocarcinoma; disease-minus-control signatures and cohort memberships.
4. Gene-program memberships, stress thresholds, profile-family assignments,
   designability weights and normalization definitions.
5. Model configurations, sampling settings, optimizer schedules, seeds,
   trained checkpoints and independent ExPO model interface/weights.
6. Candidate outputs, external experimental profiles, target-level results, and figure
   source files with scripts linking each figure or table to its inputs.
7. The frozen evaluation procedure, including target/profile IDs and independent sampling
   units, that permits comparison of results, denominators, and uncertainty with the paper.

The [public dataset guide](datasets.md) identifies source collections and access routes.
Those collections do not by themselves identify the study's selected rows or partitions.

Reported manuscript percentages are not embedded as outputs, expected test values, or
success criteria. Synthetic data is labeled in dataset provenance, run manifests and reports.
The `reproduces_manuscript` field remains false in this release.

## Interpretation boundaries

- The two learned forward models use the same training data. Different architectures
  reduce some model-specific coupling but do **not** create external experimental evidence.
- A compatibility logit is a ranking score, not a probability. Do not apply a sigmoid and
  label the result a success probability without a specified calibration task and data.
- Designability components involving exact/MOA/family recovery require evaluation labels.
  That version of designability is **retrospective** and cannot be used prospectively as
  though those labels were already known. A prospective proxy needs separate validation.
- A learned stress proxy is neither toxicity measurement nor a validated safety assay.
  Removing a stress component can remove biology of interest; report the target edit.
- Scaffold novelty is not patentability, synthesizability or therapeutic novelty.
- Gene-signature reversal is an optimization objective, not demonstrated disease treatment.

## Upstream methods

- [Chemprop directed message-passing documentation](https://chemprop.readthedocs.io/en/latest/tutorial/python/models/message_passing.html)
- [GraphINVENT implementation](https://github.com/MolecularAI/GraphINVENT)
- [FAIR4RS principles](https://doi.org/10.15497/RDA00068)

No upstream source was copied into the newly written model implementations. Dependency
use does not transfer ownership of, or license rights to, upstream datasets/checkpoints.
