# Primary and consensus ranking

The default policy keeps the independent forward predictor out of candidate selection.
For a separate consensus experiment, supply an explicit policy:

```bash
rigid rank data/study --checkpoint runs/model/model.pt --reranking configs/consensus.json --output runs/consensus.json
```

The supplied file contains illustrative starting weights, not fitted study parameters.
Tune weights on validation data, freeze the policy, and then evaluate held-out targets.
Correlations are calculated on the train-standardized gene space.

```text
score = compatibility_weight * compatibility_logit
        + consensus_weight * mean(primary_correlation, independent_correlation)
        - disagreement_weight * abs(primary_correlation - independent_correlation)
        - stress_weight * predicted_stress
        + novelty_weight * (1 - maximum_training_tanimoto)
```

`stress_weight` and `novelty_weight` inherit checkpoint settings when null. All weights
are nonnegative and every contribution is recorded separately in each candidate row.
`mode: primary` rejects nonzero consensus/disagreement weights; `mode: consensus`
requires a positive consensus or disagreement weight. Compatibility logits are unbounded,
so a weight is not a fraction of total evidence. No normalization is fitted on test scores.

Consensus runs explicitly mark the second predictor as a scoring component. It cannot
also count as an unused independent control. Validation against a separate empirical
catalog or a third untouched predictor remains necessary. The bundled forward models
are the architectures listed in [models.md](models.md); enabling consensus does not
convert either model into ExPO or establish empirical agreement.

The Python API accepts `RerankingConfig` as `RIGIDPipeline.rank(..., reranking=policy)`
or `screen_dataset(..., reranking=policy)`. Do not combine a policy with the legacy
`stress_weight`/`novelty_weight` arguments. Default primary rankings remain compatible.
