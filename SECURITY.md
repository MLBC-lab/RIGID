# Security policy

This is a pre-release research toolkit, not a hardened multi-tenant service. Version 0.3.x
is the currently maintained development series; no long-term support commitment is made.
Do not expose its file import, checkpoint loading or native-library interfaces as an
unauthenticated network service.

Report vulnerabilities privately to the research team through an existing trusted channel.
A public security contact/private advisory endpoint must be configured before publication.
Do not create a public issue containing credentials, private data or an unpatched exploit.

## Trust boundaries

- Load model bundles and native libraries only from trusted sources. Tensor-only loading
  reduces arbitrary-pickle execution risk but does not eliminate every serialization risk.
- SHA-256 sidecars detect accidental corruption; they are not authenticity signatures.
- HTTPS downloads require an expected digest and a size limit. Data licenses remain the
  user's responsibility. Archive extraction is deliberately not automated.
- Treat SMILES, HDF5 and other scientific formats as untrusted parser inputs. Keep RDKit,
  PyTorch, NumPy and HDF5 dependencies patched. Very large inputs can exhaust memory.
- Keep datasets, checkpoints, environments and credentials outside source control.
- Generated molecules and scores are research outputs, not safety or efficacy approval.

The included CI dependency audit reports known advisories. A successful audit is not proof
that the software is vulnerability-free. Review failures before a release and document any
accepted risk rather than suppressing the scanner wholesale.
