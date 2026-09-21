# FAIR4RS and open-source readiness

**FAIR4RS** adapts the Findable, Accessible, Interoperable and Reusable principles
to research software. This guide maps repository practices to those principles
and records remaining release tasks. The source uses the MIT license.

Reference: [FAIR4RS principles, Research Data Alliance](https://doi.org/10.15497/RDA00068).
The [OpenSSF Best Practices criteria](https://www.bestpractices.dev/en/criteria/0) provide
complementary software-development and security practices.

## Principle-by-principle evidence

| Principle | Local implementation/evidence | Remaining public-release action |
| --- | --- | --- |
| F1: globally unique persistent identifier | Package name, version and citation metadata | Deposit a release in an archive and obtain a DOI |
| F1.1: distinct software components identifiable | Named package/modules; optional native component; source boundaries | Assign separate persistent IDs if components are independently released |
| F1.2: versions identifiable | Semantic package version, changelog, Git-ready source, run revision metadata | Tag releases and archive each version |
| F2: rich metadata | `CITATION.cff`, `codemeta.json`, project metadata, research scope | Verify authorship/ORCIDs and add public repository/archive URLs |
| F3: metadata includes identifier | Name/version are recorded consistently | Add actual DOI/archival identifiers after deposit; none are invented |
| F4: metadata searchable/indexed | Machine-readable metadata prepared | Publish on a searchable source host and archival index |
| A1: retrievable by standard protocol | Portable source distribution and wheels; HTTPS acquisition utility | Publish release artifacts over HTTPS |
| A1.1: open/free protocol | HTTPS and ordinary Git distribution supported | Configure the actual public endpoint |
| A1.2: authentication where needed | Licensed input data can be prepared locally; credentials excluded | Document access controls for restricted datasets without redistributing them |
| A2: metadata persists if software unavailable | Standalone citation, metadata and provenance files | Deposit durable metadata in the archive |
| I1: community-standard exchange | CSV, JSON, SMILES, GCT 1.2/GCTX; ordered gene manifests; clear units | Verify exact community identifiers for the experimental datasets |
| I2: qualified references to other objects | Input checksums/source records, method references, dependency metadata | Add stable dataset/checkpoint/accession references when obtained |
| R1: clear descriptive attributes | README, API documentation, scientific boundaries, examples, configurations | Keep documentation synchronized with validated release results |
| R1.1: clear license | `LICENSE` (MIT); dataset licenses recorded separately | Confirm rights and lab/institutional approval before publication |
| R1.2: detailed provenance | Dataset transformations, source hashes, run configuration, versions and fit IDs | Attach experimental provenance and archive releases |
| R2: qualified dependencies | Version-constrained `pyproject.toml`, tested local dependency snapshot | Test supported version ranges and publish lock/SBOM artifacts per release |
| R3: domain community standards | Chemistry standardization, leakage checks, partition discipline, numerical tests | Review domain conventions against the study protocol |

FAIR4RS compliance cannot be completed solely by generating a local directory. The missing
hosting, indexing and persistent identifiers are operational tasks; this document makes
those gaps explicit. There is no FAIR certification or passing badge claimed here.

## OpenSSF-oriented practices included

- Public-facing contribution, conduct, security and support policies.
- Automated testing, lint/format checks, packaging and optional native-kernel tests.
- Least-privilege CI permissions; pull-request CI has no publishing secrets.
- Explicit dependency metadata, dependency-update configuration and vulnerability-audit job.
- Strict data/configuration validation; tensor-only checkpoint loading; checksum integrity.
- Epoch-boundary recovery with source/data/configuration guards, bounded cache diagnostics,
  data-quality reports, explicit run comparison and installed-wheel smoke testing.
- A changelog, proposed release checklist, and clear supported-version/security scope.

Repository hosting, protected branches, review requirements, maintainer responsiveness,
signed releases, vulnerability handling in practice, and a stable test history still need
to be established. Do not display an OpenSSF passing badge until the actual project meets
the criteria and completes the applicable assessment.

## Before sharing a public release

1. Confirm the research scope, author list, institutional rights and MIT license.
2. Resolve experimental data permissions and remove private artifacts from the release.
3. Run all tests on supported operating systems/Python versions, including native parity.
4. Review dependency vulnerabilities, generate a dependency inventory/SBOM, and record results.
5. Create the actual source-host repository, review settings and security contact channel.
6. Tag a version, build source/wheel distributions, publish checksums and sign artifacts.
7. Archive that version, obtain its DOI, and update citation/CodeMeta with real identifiers.
8. Publish benchmark provenance separately from synthetic smoke tests.
