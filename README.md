# data-extraction-evaluation-toolkit

A suite of tools, data models, etc. for extracting data from documents (e.g. papers) and evaluating the performance of such extraction tasks.

## tl, dr

[Look here if you just want help running deet via the CLI](#using-deet-via-the-cli).

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

### Using `deet` via the CLI

The deet command line interface (CLI) allows you to use deet without writing any python code.

#### Getting started

Once you have installed DEET, and activated your environment, (see [installation](#installation)), you should be able to run the CLI.

To test this run

```bash
deet --help
```

You should see a list of commands. To get more information on any of these (for example import-gold-standard-data), run

```bash
deet import-gold-standard-data --help
```

You can call these commands from anywhere where your virtual environment with deet installed is activated. It's a good idea to navigate into a new directory for each data extraction project, where you can store the outputs of running deet for your project.

To navigate to another directory use the `cd` command. `cd ..` will move up a level, cd `deet-project` will move into the `deet-project` folder, assuming it exists in the current directory. Type `ls` to see the files and folders in your current directory.

#### Importing and linking data

Most commands start by importing gold standard data, and require that you specify the location of a file that contains:

- bibliographic records of documents,
- the attributes that should be extracted from them, and
- (optionally) the gold standard annotations of those documents made by humans

DEET currently supports importing this data from an EPPIJson file.

If you want to extract data from the full text of documents, you will need to link your documents to those full texts.

Assuming you have an EPPIJson file called `references.json` in your current working directory, and a folder of PDFs for those documents in a directory called `pdfs`, you can link these together with

```bash
deet link-documents-fulltexts references.json
```

You can specify the pdf folder to look through (which does not have to be in your current directory) by setting the option --pdf-dir, but this defaults to `pdfs`.

Deet will try to link documents to pdfs using IDs, or a combination of author and year, but if you want to specify a file containing a map between your EPPIJson document_ids and the filenames of your pdfs, you can create a template for this with the `deet init-linkage-mapping-file` command. Once you have filled the template by specifying the file name for each document, you can link documents by pointing to this file

```bash
deet link-documents-fulltexts --link-map-path link_map.csv references.json
```

Linking will parse the pdfs, and save the contents (along with the bibliographic records to wherever is specified in the option `--output-path`. This defaults to `linked_documents`

#### Extracting data from parsed/linked data

Once you have linked this data to pdfs if you are doing full text data extraction, you are ready to extract data from them.

Use the command `deet extract-data` to extract data from imported documents.

Once again, you will need to specify an EPPIJson file you want to import from, e.g.

`deet extract-data references.json`

By default, extract-data will try to extract attributes using prompts specified in the EPPIJson file.

If you want to edit the prompts used for data extraction, you can do this by setting the `--prompt-population` option to `cli`, to fill in prompts in the command line, or by setting `--prompt-population` to `file`, and pointing to a csv detailing a prompt for each attribute with `--csv-path`. To create a template for this csv, run `deet write-prompt-csv`.

To set further configuration options, you can supply a path to a configuration file with the option `--config-path`. To create a template for this file detailing configurable options, run

```bash
deet export-default-config
```

You can edit this file to change the configuration options for your extract-data pipeline.

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
