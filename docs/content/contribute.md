# Contributor guidelines

Contributions are welcome. Please follow these guidelines:

- Use Python 3.13–3.14 and [Poetry](https://python-poetry.org/) 2.5.1.
  Run `poetry install` to install the locked dependencies and
  `poetry run pytest` to run the tests and guide doctests.
- Use dependency groups: `poetry add <package> --group dev` adds a development
  dependency; `poetry add <package>` adds a runtime dependency. The project uses
  `dev`, `tests`, `docs`, and `interactive` groups.
- Follow the Black code style.
- Annotate function parameters and returns, including private numerical helpers.
  Match the actual scalar, array, and container types; do not change model fields
  merely to satisfy typing.
- Document public APIs with NumPy-style docstrings and Markdown body content.
  Describe array shapes, standardized versus physical parameter coordinates,
  covariance factors, mutation, and meaningful errors. Give private helpers
  concise contracts rather than repetitive boilerplate.
- Run `poetry run pyright` to check the library against the minimum Python
  version. CI also checks each supported Python version.
- Add behavioral regression tests for fixes. Run `poetry run pytest` for tests
  and runnable documentation examples.
- Prefer semantic commit messages: `feat`, `fix`, `docs`, `refactor`, `test`,
  `chore`, or `ci`.
- Run `poetry run ruff check .` to lint, adding `--fix` for supported fixes.

## Documentation

The site uses [Zensical](https://zensical.org/), Markdown guides, and
[mkdocstrings](https://mkdocstrings.github.io/) for the live Python reference.

```sh
poetry install --only main,docs
poetry run zensical serve
```

Create a production build with `poetry run zensical build --strict`. Output goes
to `site/`; GitHub Pages publishes that directory. No Sphinx or system Pandoc
installation is needed.

Edit Markdown directly. Preserve the Python-console examples and leave a blank
line before their closing code fence so the standard doctest parser can read
expected output correctly. Run `poetry run pytest --no-cov docs` to check them.

The cookbook renders `.ipynb` files through `nbconvert` without executing them.
Update and save notebook outputs when changing an example; the site includes
those outputs and a download link to the original notebook. New notebook pages
need a Markdown wrapper like the existing examples and an entry in
`zensical.toml`.

The API uses NumPy-style docstrings. `docs_api.py` supplies the runtime-generated
constructor signatures used by Pydantic and the prior metaclass; other API data
is collected directly from Python source. Keep constructor documentation with
the class and mathematical expressions in Markdown/LaTeX.

## License

This project is licensed under the terms of the MIT license.

## Code of conduct

This project follows the [Contributor Covenant](../code_of_conduct.md).
