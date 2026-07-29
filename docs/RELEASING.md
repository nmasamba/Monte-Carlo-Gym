# Releasing MonteCarloGym to PyPI

PyPI distributions contain the dependency-light classical kernel, adaptive
protocols, and experiment utilities. Gymnasium remains the optional `gym`
extra; no neural framework is required for PUCT because predictors are injected.

## Preconditions

1. Choose a unique PEP 440 version in both `pyproject.toml` and
   `src/montecarlgym/__init__.py`. PyPI never permits replacing a file for a
   version that has already been uploaded.
2. Ensure the intended package name is available or owned by the release team.
   The official PyPI and TestPyPI JSON endpoints both returned `404` for
   `montecarlgym` on 2026-07-29, but name availability can change before the
   first upload, so recheck immediately before release.
3. Run the full tests, experiment smoke, examples, and isolated wheel test.
4. Commit the release, merge it to `main`, and create a signed or annotated tag,
   for example `v0.2.0a2`.

## Local build and TestPyPI rehearsal

Use a clean virtual environment:

```bash
python -m venv .venv-release
. .venv-release/bin/activate
python -m pip install --upgrade pip
python -m pip install '.[gym,release]'

python -m unittest discover -s tests -v
python -m build
python -m twine check dist/*
```

Create a separate TestPyPI account/token, then upload:

```bash
python -m twine upload \
  --repository-url https://test.pypi.org/legacy/ \
  dist/*
```

Test the exact wheel from TestPyPI while resolving optional dependencies from
the production index. Keep TestPyPI isolated from dependency resolution:

```bash
python -m venv .venv-test-install
. .venv-test-install/bin/activate
python -m pip install 'gymnasium>=1.0'
python -m pip install --no-deps \
  --index-url https://test.pypi.org/simple/ \
  'montecarlgym==0.2.0a2'
python -c 'import montecarlgym; print(montecarlgym.__version__)'
```

## Recommended production publication: Trusted Publishing

The repository includes `.github/workflows/publish.yml`. In the PyPI project:

1. Add a GitHub Trusted Publisher for owner `nmasamba`, repository
   `Monte-Carlo-Gym`, workflow `publish.yml`, and environment `pypi`.
2. In GitHub, create the `pypi` deployment environment and protect it with
   required reviewers if desired.
3. Publish a GitHub release for the matching version tag. The workflow rebuilds
   the distributions, runs the unit suite and `twine check`, and publishes them
   using short-lived OpenID Connect credentials; no long-lived PyPI token is
   stored in GitHub.

For the first-ever upload, PyPI supports a pending Trusted Publisher configured
before the project exists. Alternatively, a project owner can perform the first
production upload manually:

```bash
python -m twine upload dist/*
```

After publication, verify from a new environment:

```bash
python -m venv .venv-pypi-check
. .venv-pypi-check/bin/activate
python -m pip install --no-cache-dir 'montecarlgym[gym]==0.2.0a2'
python -c 'import montecarlgym; print(montecarlgym.__version__)'
```

Do not upload from a dirty worktree, reuse a version, or publish synthetic toy
measurements as research results.
