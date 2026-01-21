# data-extraction-evaluation-toolkit

A suite of tools, data models, etc. for extracting data from documents (e.g. papers) and evaluating the performance of such extraction tasks.

## tl, dr

A key innovation the Destiny project seeks to deliver is a toolkit for automating the extraction of attributes of interest from documents (e.g. academic papers). This way, large repositories of published research can have relevant data extracted to use for evidence synthesis, thereby freeing up researchers to dedicate time and resources to higher-value tasks.

This software enables this end-to-end process for data extraction and evaluation tasks. `data-extraction-evaluation-toolkit` is conceived of as a modular suite of tools, allowing users to include and exclude specific modules in line with their needs. For instance, while you may want to supply a pdf and extract structured information from it, you may have already parsed pdfs, or other file sources into a more processing-friendly format (markdown), and hence choose to omit the parser module from your data extraction pipeline.

## Installation

### Installing `pandoc`

This software depends on [`pandoc`](https://pandoc.org/), a widespread open source file/document conversion utility. In this codebase, this is implemented using `pypandoc`, but this depends on `pandoc` being installed at a system level.

```shell
brew install pandoc # mac
apt install pandoc # ubuntu/debian
choco install pandoc # windows
```

Alternative installation instructions for Windows users are [shown further below](###pandoc)
### Installing `uv`

[uv](https://docs.astral.sh/uv) is used for dependency management and managing virtual environments. You can install uv either using pipx or the uv installer script:

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Installing Dependencies

Once uv is installed, install dependencies:

```sh
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

## Using `DEET`

The `data-extraction-evaluation-toolkit` (`DEET`) contains mutliple modules which can be leveraged alone, or orchestrated together to form a `Pipeline`. The goal of `DEET` is to be modular and extensible, allowing users to customise a specific pipeline or workflow to their needs.

Currently, the app covers the following tools:

- **Document parsing** (from a range of formats; typically into `markdown`)
- **Gold standard data ingestion and standardisation** (currently only `EPPI-json` datasets are supported)
- **LLM-powered data extraction**
- **Orchetration of tools into `Pipeline`s** (these tools can be existing `DEET` modules, custom python functions, or scripts (`R`, `python`, `bash` currently suppported.))

Our roadmap for future development contains:

- **Comparison & evaluation of LLM vs human annotations**
- **Support for LLM prompts and SQL code**

## Document parsing

The first time you run anything from `deet/parser.py`, you will likely have to wait for a considerable (5-15 minutes) amount of time, as dependencies will collect and install. These dependencies include machine learning libraries and pre-trained models.

## Data Processing

### Annotation Converter

The annotation converter can be used to convert raw EPPI-Reviewer data into structured format.

**Usage:**

```bash
uv run python deet/scripts/annotation_converter_cli.py <input_file> <output_dir>
```

**Example:**

```bash
uv run python deet/scripts/annotation_converter_cli.py deet/annotations/raw/eppi/sample_eppi.json output/processed
```

This creates an organized directory structure:

```text
output/processed/eppi/{filename_without_extension}/
├── attributes.json
├── documents.json
├── annotated_documents.json
└── attribute_id_to_label_mapping.json
```

For example, processing `sample_eppi.json` creates `output/processed/eppi/sample_eppi/` with the JSON files inside.

## LLM-powered data extraction

## Orchastration into `Pipeline`s

## Contributing

If you want to contribute to this project -- awesome, everyone's welcome.
Please check existing issues, and perhaps pick something from there? Please create a new branch off `development`.
Once you're ready for your code to be reviewed, submit a PR into `development.

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

- https://github.com/jgm/pandoc/releases

Select the `.msi` file from the latest release (e.g. `pandoc-3.8.3-windows-x86_64.msi` for release 3.8.3, as of January 2026) and run the installer by double-clicking on it.

Alternatively, one may download the ZIP file from the link above, unpack it, and move the binaries to a directory of your choice (not tested; more information is available via the link below).

For other options (Chocolatey or Winget package manager, or Conda Forge), see:

- https://pandoc.org/installing.html

**Note:** Using the `.msi` method should remove older versions of Pandoc and update PATH variables automatically, while other methods may cause multiple versions to be installed.

---

#### 2. Verifying that Pandoc works

These instructions are taken from:
- https://pandoc.org/getting-started.html

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
   _Copyright (C) 2006-2025 John MacFarlane. Web: https://pandoc.org_
   
   _This is free software; see the source for copying conditions. There is no_
   _warranty, not even for merchantability or fitness for a particular purpose._

4. Navigate to a directory of your choice using the `cd` command.  
   See: https://stackoverflow.com/questions/17753986/how-to-change-directory-using-windows-command-line  
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
   - `-s` creates a *standalone* file with a header and footer.  
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
