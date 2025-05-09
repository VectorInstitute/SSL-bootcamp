# Contributing to AI Engineering Projects

Thanks for your interest in contributing!

To submit PRs, please fill out the PR template along with the PR. If the PR
fixes an issue, don't forget to link the PR to the issue!

## Pre-commit hooks

Once the python virtual environment is setup, you can run pre-commit hooks using:

```bash
pre-commit run --all-files
```

## Coding guidelines

For code style, we recommend the [google style guide](https://google.github.io/styleguide/pyguide.html).

Pre-commit hooks apply the [black](https://black.readthedocs.io/en/stable/the_black_code_style/current_style.html)
code formatting.

For docstrings we use [numpy format](https://numpydoc.readthedocs.io/en/latest/format.html).

We also use [ruff](https://github.com/astral-sh/ruff) for further static code analysis.
The pre-commit hooks show errors which you need to fix before submitting a PR.

Last but not the least, we use type hints in our code which is then checked using
[mypy](https://mypy.readthedocs.io/en/stable/). Currently, mypy checks are not
strict, but will be enforced more as the API code becomes more stable.
