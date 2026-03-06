# data-extraction-evaluation-toolkit

A suite of tools, data models, etc. for extracting data from documents (e.g. papers) and evaluating the performance of such extraction tasks.

## tl, dr

[Look here if you just want help running default pipelines](#default-pipelines).

A key innovation of the [Destiny project](https://destiny-evidence.github.io/website/) is a toolkit for automating the extraction of attributes of interest from documents (e.g. academic papers). This way, large repositories of published research can have relevant data extracted to use for evidence synthesis, thereby freeing up researchers to dedicate time and resources to higher-value tasks.

This software enables this end-to-end process for data extraction and evaluation tasks. **`data-extraction-evaluation-toolkit`**; or **`deet`** is conceived of as a modular suite of tools, allowing users to include and exclude specific modules in line with their needs. For instance, while you may want to supply a pdf and extract structured information from it, you may have already parsed pdfs, or other file sources into a more processing-friendly format (markdown), and hence choose to omit the parser module from your data extraction pipeline.

## Installation

### Installing `pandoc`

This software depends on [`pandoc`](https://pandoc.org/), a widespread open source file/document conversion utility. In this codebase, this is implemented using `pypandoc`, but this depends on `pandoc` being installed at a system level.

```shell
brew install pandoc # mac
apt install pandoc # ubuntu/debian
choco install pandoc # windows
```

Alternative installation instructions for Windows users are [shown further below](#pandoc)

### Installing `uv`

[uv](https://docs.astral.sh/uv) is used for dependency management and managing virtual environments. You can install uv either using pipx or the uv installer script:

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Installing Dependencies

Once uv is installed, install dependencies:

```sh
git clone git@github.com:destiny-evidence/data-extraction-evaluation-toolkit.git # SSH
# or
git clone https://github.com/destiny-evidence/data-extraction-evaluation-toolkit.git # HTTPS
cd data-extraction-evaluation-toolkit
uv sync
```

### Activate your environment

```sh
source .venv/bin/activate
```

### Installing pre-commit hooks

Install `pre-commit` locally (in your activated `venv`) to aid code consistency (if you're looking to contribute).

```sh
pre-commit install
```

## Using `deet`

The `data-extraction-evaluation-toolkit` (`deet`) contains mutliple modules which can be leveraged alone, or orchestrated together to form a `Pipeline`. The goal of `DEET` is to be modular and extensible, allowing users to customise a specific pipeline or workflow to their needs.

Currently, the app covers the following tools:

- **Document parsing** (from a range of formats; typically into `markdown`)
- **Gold standard data ingestion and standardisation** (currently only `eppi.json` datasets are supported out of the box, for other datasets, use the data models in `data_models/base.py` to ingest your gold standard references.)
- **LLM-powered data extraction**
- **Orchetration of tools into `Pipeline`s** (these tools can be existing `DEET` modules, custom python functions, or scripts (`R`, `python`, `bash` currently suppported.))

Our roadmap for future development contains:

- **Linking of gold standard references & pdf-derived parsed documents**
- **A fully-fledged cli for typical `deet` tasks**
- **A framework for repeatable pipeline runs with slight modifications for comparison**
- **Comparison & evaluation of LLM vs human annotations**
- **Support for prompt versioning tool**

### Default pipelines

Default pipelines leverage the included modules to chain typical, often-used data tasks.
Currently, they are implemented as scripts, but will soon be moved into a `cli`.
See available scripts prefixed with `pipeline_` in `deet/scripts/`.
All these scripts will require:

- a path to a directory of input files, `.md` or `.pdf`
- a path to a `eppi.json`
- paths to other relevant files, e.g. csvs containing prompts.
These scripts will then produce a file `llm_extractions.json` containing llm classifications for each of the files contained in the directory.

So, you might want to run something like this:

```python
python deet/scripts/pipeline_interactive_prompt_generation.py -m path/to/my/markdown/files -e path/to/eppi.json -o path/to/write/out/files
```

If you're unsure what kind of arguments a given script might required, you can always run something with the `-h` or `--help` flag, to get more info:

```python
python deet/scripts/pipeline_interactive_prompt_generation.py -h

usage: pipeline_interactive_prompt_population.py [-h] [-p PDF_PATH] [-m MARKDOWN_PATH] -e EPPI_JSON_PATH
                                                 [-f FILTER_ATTRIBUTE_IDS [FILTER_ATTRIBUTE_IDS ...]] [-o OUTPUT_PATH]

options:
  -h, --help            show this help message and exit
  -p PDF_PATH, --pdf_path PDF_PATH
                        directory containing PDF files
  -m MARKDOWN_PATH, --markdown_path MARKDOWN_PATH
                        directory containing or for markdown files
  -e EPPI_JSON_PATH, --eppi_json_path EPPI_JSON_PATH
                        path to eppi json
  -f FILTER_ATTRIBUTE_IDS [FILTER_ATTRIBUTE_IDS ...], --filter_attribute_ids FILTER_ATTRIBUTE_IDS [FILTER_ATTRIBUTE_IDS ...]
                        an optional list of attribute_ids to filter by.
  -o OUTPUT_PATH, --output_path OUTPUT_PATH
                        path to save output JSON (auto-generated if not provided)
```

## Contributing

If you want to contribute to this project -- awesome, everyone's welcome.
Please see the [contributing guidelines](CONTRIBUTING.md) for details on how best to contribute.

## Tests

Tests are written using `pytest`. You can run the tests locally using

```shell
pytest
```

Unit tests are automatically run in [Continuous Integration](https://en.m.wikipedia.org/wiki/Continuous_integration) (CI) using github actions (see `.github/workflows/tests.yml`) on Pull Requests or merges into `main` or `development`. Integration tests are also run for pushes/PRs into `main` (Note: these will take approx 1-2h to complete, so consider a cup of coffee while you wait).

## Platform-specific installation instructions

### PANDOC

#### 1. Installation

There are multiple options. The first and possibly best one is using an installer file provided via GitHub. For the most recent release, visit:

- <https://github.com/jgm/pandoc/releases>

Select the `.msi` file from the latest release (e.g. `pandoc-3.8.3-windows-x86_64.msi` for release 3.8.3, as of January 2026) and run the installer by double-clicking on it.

Alternatively, one may download the ZIP file from the link above, unpack it, and move the binaries to a directory of your choice (not tested; more information is available via the link below).

For other options (Chocolatey or Winget package manager, or Conda Forge), see:

- <https://pandoc.org/installing.html>

**Note:** Using the `.msi` method should remove older versions of Pandoc and update PATH variables automatically, while other methods may cause multiple versions to be installed.

---

#### 2. Verifying that Pandoc works

These instructions are taken from:

- <https://pandoc.org/getting-started.html>

##### Steps

1. Search for `cmd` in the Windows Start menu to launch **Command Prompt**. It should also be possible to use **Windows PowerShell**.
2. If you are using `cmd`, type the following before using Pandoc to set the encoding to UTF-8:

   ```batch
   chcp 65001
   ```

3. Type the following and press Enter to check if Pandoc is installed:

   ```batch
   pandoc --version
   ```

   It should say something like:

   _pandoc 3.8.3_
   _Features: +server +lua_
   _Scripting engine: Lua 5.4_
   _User data directory: C:\Users\YourUserName\AppData\Roaming\pandoc_
   _Copyright (C) 2006-2025 John MacFarlane. Web: <https://pandoc.org>_

   _This is free software; see the source for copying conditions. There is no_
   _warranty, not even for merchantability or fitness for a particular purpose._

4. Navigate to a directory of your choice using the `cd` command.
   See: <https://stackoverflow.com/questions/17753986/how-to-change-directory-using-windows-command-line>
   (This link is useful if you need to switch to a different drive.)

5. Create a new directory to test Pandoc:

    ```batch
   mkdir pandoc-test
   ```

6. Navigate into the new directory:

    ```batch
   cd pandoc-test
   ```

   It should now show `pandoc-test` before the blinking cursor in the command prompt.
   If the directory is not shown, run:

    ```batch
   echo %cd%
   ```

7. Outside the command prompt (using Notepad, for example), create a file called `test1.md` in your `pandoc-test` directory. If Windows is hiding file extensions then please make sure that the displayed file name corresponds to the actual file name, and that no hidden  '.txt' extension is added. Paste the following text into your file, then save the file.

   ```text
   ---

   title: Test
   ...

   # Test!

   This is a test of *pandoc*.

   - list one
   - list two

   ```

8. Back in the command prompt (still in the `pandoc-test` directory), type  `dir` and press Enter. You should see `test1.md` listed.

9. Convert the file to HTML by typing:

   ```batch
   pandoc test1.md -f markdown -t html -s -o test1.html
   ```

   **Info:**
   - `test1.md` tells Pandoc which file to convert.
   - `-s` creates a _standalone_ file with a header and footer.
   - `-o test1.html` specifies the output file name.

   Note that `-f markdown` and `-t html` could be omitted, since the default is to convert from Markdown to HTML.

10. Type the following to open the HTML file in your browser:

    ```batch
    .\test1.html
    ```

    Alternatively, open the folder in File Explorer and double-click `test1.html`.

   ---

#### 3. Completion

   If the HTML file opens and displays well then you have successfully installed Pandoc and converted a Markdown file to HTML.
Unit tests are automatically run in [Continuous Integration](https://en.m.wikipedia.org/wiki/Continuous_integration) (CI) using github actions (see `.github/workflows/tests.yml`) on Pull Requests or merges into `main` or `development`.
