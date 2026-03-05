# data-extraction-evaluation-toolkit

A suite of tools, data models, etc. for extracting data from documents (e.g. papers) and evaluating the performance of such extraction tasks.

## Adding documentation

Please add to the docs whenever you feel it would be useful. The docs are built using [mkdocs](https://www.mkdocs.org/) and mix automatically-generated API documentation with more general documentation. An automatically generated html static site is built from the `docs/` directory, and the API documentation is generated from docstrings in the code.

To add your own documentation, add markdown files to the `docs/` directory _and_ add these to the `nav` block in `mkdocs.yml`. To add API documentation, add docstrings to the code and ensure that the relevant modules are included in the `nav` block in `mkdocs.yml`.

To build the docs locally, run `mkdocs build --strict` from the root of the repository and open `site/index.html` in a browser. The documentation website is currently automatically built and deployed to GitHub Pages on pushes to the `main` branch, and uses the `gh-pages` branch to serve the docs.

The documentation website is available at [https://destiny-evidence.github.io/deet](https://destiny-evidence.github.io/deet).
