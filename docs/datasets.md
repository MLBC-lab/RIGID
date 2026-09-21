# Public datasets and study-specific inputs

RIGID imports compound-response expression matrices and explicit profile/compound
metadata. The collections below are supported public input sources, not a declaration
of the exact releases or rows used in the manuscript.

## Perturbational expression resources

| Resource | Access | Use in a RIGID workflow |
| --- | --- | --- |
| LINCS L1000 Phase I | [GEO GSE92742](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE92742) | Expression files and associated metadata for constructing a declared compound-response dataset |
| LINCS L1000 Phase II | [GEO GSE70138](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE70138) | A separately versioned source collection; align identifiers and processing before combining releases |
| Broad GEO guide | [L1000 files on GEO](https://clue.io/geo-guide) | Guidance on locating and interpreting the Phase I and Phase II files |

Phase I distributes multiple processing levels, including levels 1–5. Choose the
matrix level explicitly and retain its matching metadata; processing levels are not
interchangeable representations of the same training input. See the
[GSE92742 record](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE92742).

Use GEO for durable access to these public collections. The
[CLUE data page](https://clue.io/data) announces platform retirement effective
January 31, 2026 and directs users to GEO for public data. This workflow therefore
does not depend on a CLUE account or API.

## Disease cohorts and gene programs

[NCBI GEO](https://www.ncbi.nlm.nih.gov/geo/) is a starting point for locating
disease-expression cohorts. No accession is assigned here to the psoriasis,
ER-positive breast cancer, or lung adenocarcinoma analyses: a disease label alone
does not establish which participants, controls, assay, or processing were used.

For a disease-minus-control target, record the cohort accession, selected sample IDs,
case/control definition, covariates, assay preprocessing, gene mapping, contrast,
and aggregation rule. Preserve separate cohort identities when evaluating stability.
Gene-program and stress definitions must include membership, identifier namespace,
source/version, scoring rule, and any threshold.

## Prepare an auditable input

1. Select the release and processing level; record exact filenames, source URLs,
   download dates, checksums, and applicable data-use terms.
2. Create the ordered landmark-gene manifest. Verify identifiers and order against
   the selected source metadata; a list containing 978 entries is not sufficient.
3. Join expression profiles to perturbagen identity, structure, cell, dose, and time
   by stable identifiers. Retain inclusion/exclusion reasons and replicate handling.
4. Save target IDs, reference-catalog IDs, and split assignments before evaluating
   outcomes. A public source collection does not specify an experiment's holdouts.
5. Import with the [data preparation workflow](data.md), run quality/leakage checks,
   and preserve the generated provenance alongside each result.

Downloading a whole GEO series does not establish the study's precise subset or
evaluation protocol. Contact the manuscript's corresponding authors or repository
maintainer to request the [study materials](SCIENTIFIC_CONTRACT.md), including
target/profile IDs, split manifests, configurations, weights, program sets, figure
sources, and target-level results. These are not bundled; availability and permission
to redistribute them must be confirmed with the provider.
