# Training execution and recovery

The ranking bundle, graph generator, and fingerprint VAE share one optimization
engine. Model-specific code supplies the objective; the engine owns batching,
AdamW, gradient clipping, validation, early stopping, and recovery.

## Resume a run

Repeat the original training command with `--resume`. Keep its dataset, settings,
and model options unchanged:

```sh
rigid train data/curated --config configs/research.json --output runs/ranker --scaffold-holdout
rigid train data/curated --config configs/research.json --output runs/ranker --scaffold-holdout --resume
```

The same option is available for `train-generator` and `train-vae`. The Python
entry points accept the keyword `resume=True`. Existing output directories are
still rejected by default. Resuming an already completed run verifies its final
checkpoint and returns the existing manifest without retraining or overwriting
the model.

The ranking workflow has three phases: `forward`, `independent`, and `ranker`.
Completed phases restore their selected model and RNG progression; only the
unfinished phase continues. This matters when the ranker is initialized after
the two response predictors.

## Recovery boundary

Checkpoints represent **completed epochs**, not individual batches. Every
committed state contains:

- Current model and optimizer tensors, together with the best validation model.
- Completed epoch count, validation minimum, patience counter, and full history.
- Python, NumPy, Torch, device, and permutation-generator RNG states.
- A fingerprint of the data, objective, architecture, partitions, and settings.

An interrupted epoch is repeated from its starting boundary. A batch that changed
parameters before interruption is not counted twice. Cancellation and failure
restore that boundary in memory and retain the durable state on disk. The first
snapshot is taken before any optimizer updates, so failures during epoch one are
also recoverable.

On each commit, an immutable tensor-only checkpoint and its checksum are written
before `latest.json` is atomically replaced. The active and immediately preceding
snapshot are retained. An incomplete, unreferenced snapshot is never selected
implicitly. Checksum failures stop the run; recovery does not silently substitute
an older epoch. The standard exported `model.pt` remains immutable.

## Run artifacts

```text
run/
├── training-session.json        # Full protocol and fingerprint
├── optimization/
│   └── <phase>/
│       ├── latest.json         # Active snapshot and checksum
│       ├── state-*.pt          # Active and previous optimizer generations
│       ├── events.jsonl        # Append-only lifecycle and epoch records
│       ├── heartbeat.json      # Most recent progress or failure event
│       └── .writer.lock        # OS-managed single-writer lock
├── history.json                # Exported history after successful training
├── manifest.json               # Data, model-selection and execution provenance
└── model.pt                    # Best-validation models for inference
```

The lock is released by the operating system when its process exits. The lock
file may remain and is not an indication of an active writer. Each phase rejects
a second simultaneous writer.

Events include phase, UTC timestamp, completed epoch count, and an event name.
Completed-epoch records include training loss, validation loss, and elapsed
seconds. Long epochs emit progress at batch boundaries at least every 30 seconds
when control returns to the engine. A single long model operation can delay that
heartbeat. `cancelled` and `failed` events report the exception type and message;
the checkpoint pointer remains the authority for durable progress. A stale
heartbeat alone cannot distinguish a slow operation from a killed process.

## Protocol checks

Public trainers reject changes to input file checksums, all configuration fields,
gene-program options, generator/VAE options, scaffold policy, package versions,
platform, and implementation source checksums. The engine additionally records
model structure and trainability, partition membership and order, epoch budget,
and active determinism/thread settings. Moving byte-identical inputs does not
change their identity. Even a change only to held-out data is rejected for resume,
since the run's recorded dataset would otherwise become ambiguous.

Changing the epoch budget is a new experiment, not a resume. Fine-tuning and
cross-version migration are separate workflows and are not implemented by this
API. The fingerprint is an integrity and compatibility check, not a digital
signature or protection against someone rewriting every local artifact.

## Scientific and numerical boundaries

Only declared training indices enter gradient updates. Validation indices must be
nonempty, unique, and disjoint from training; validation runs in evaluation mode
under `torch.no_grad()`, after training gradients have been cleared. Test and
external partitions are not accepted by the public training objective builders.
Their file bytes appear in provenance checksums, not in training gradients or
validation selection.

Losses must be finite scalar tensors. The engine rejects disconnected objectives,
nonfinite gradients, nonfinite parameter updates, and nonfinite aggregated epoch
losses. Checkpoint selection still uses the original minimum validation loss and
patience rule. No new calibration or success-probability claim is introduced.

CPU tests compare uninterrupted and interrupted/resumed runs bit for bit,
including dropout, stochastic latent sampling, Python/NumPy/Torch randomness,
early-stopping state, and transitions between all three ranking phases. Wall-clock
timings are expected to differ. Identical results are not promised across hardware
or library versions; nondeterministic device kernels remain a separate concern.
This is a single-process engine, not a distributed trainer.

## Custom objectives

`rigid.engine.fit_model` accepts a typed loss callback with signature
`loss(indices, is_training, epoch) -> scalar Tensor`. Supply `state_directory`
and an explicit JSON-serializable `protocol` describing the input identity and
objective to enable durable recovery. `should_cancel` supports cooperative
cancellation and `on_epoch` receives an observational copy of each committed epoch
record.

Callbacks must not mutate external state or use private, unrecorded RNGs. Derive
such sampling deterministically from the supplied epoch and indices, or include
it in the objective's explicit state management. Observers must not modify the
model or consume RNG state. The built-in trainers follow these constraints.

Resume requires byte-identical input data and a compatible source tree. Keep the
optimization snapshots with the run's provenance when archiving a completed experiment.
