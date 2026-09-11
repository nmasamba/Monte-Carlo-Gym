# Confirmatory experiments

**Status (2026-09-10):** no confirmatory study or result exists.

This directory contains documentation and frozen analysis entry points only.
Raw results are written to:

```text
output/confirmatory/<study-id>/<run-id>/
```

The confirmatory runner accepts a frozen preregistration manifest - not a mutable
config—and refuses non-empty output directories. It verifies the embedded
protocol fingerprint, repository-local artifact hashes, clean source revision,
and allowed manifest-only revision delta before running. Do not place
exploratory runs here, and do not create placeholder outcome files before the
registered run.

The current SQLite mechanism candidate is not eligible for this workflow. A
fresh sequential-agent protocol is required after the gates in
`docs/RESEARCH_STATUS.md` pass.
