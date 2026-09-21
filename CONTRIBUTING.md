# Contributing

Use an issue or discussion on the project's eventual public source host to describe bugs,
propose an experiment or discuss an API change. Until a public host is established, share
patches with the research team directly. Do not upload private datasets or credentials.

1. Create a focused branch and explain the expected scientific/engineering behavior.
2. Install the development extras and run the test/lint/format commands in the README.
3. Add regression tests. Changes to splits, features, metrics or chemical standardization
   require explicit before/after examples and an updated scientific contract.
4. Include configuration, software versions, input provenance and a minimal synthetic
   reproducer for numerical issues. Never substitute reported paper metrics for computation.
5. Update relevant documentation and the changelog. Keep comments concise and explain
   non-obvious scientific choices or invariants, not each line of syntax.

Submissions are reviewed for correctness, reproducibility, clarity and maintainability.
Unnecessary abstraction or complexity is not a substitute for evidence. Contributions
are accepted under the project's MIT license; contributors retain copyright in their work.
Do not contribute code or data you are not authorized to redistribute.

The public maintainer roster, support channel, review requirements and response targets
will be established when the research team publishes the repository.
