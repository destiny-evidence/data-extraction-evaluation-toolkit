# Installation

This page shows different ways to install `deet` across platforms and use-cases

## Basic setup and dependencies

### Accessing the terminal

=== "Windows"
    Press the windows key, type "powershell", and press enter. This is your command line.
=== "Mac/Linux"
    If you a mac user, open the terminal from `Applications` -> `Utilities` -> `Terminal`.

    If you are Linux user, press `Ctrl` + `Shift` + `t`

### Installing a package manager

=== "Windows"
    A package manager is a tool that makes it easier to install other command line tools. You could use either [scoop](https://scoop.sh/) or [cholatey](https://chocolatey.org/).

    We recommend scoop, as it does not require admin privileges to install packages. If you prefer chocolatey, replace `scoop install X` with `choco install X` in the following commands.

=== "Mac/Linux"
    Mac/linux systems come with package managers included. For mac users, this is `brew`, for (Debian) linux users, this is `apt`.

### Installing pandoc

If you want to use deet with full-texts pdfs, as opposed to abstracts, or pre-processed markdown files, you will need to have installed [`pandoc`](https://pandoc.org/), a widespread open source file/document conversion utility.

=== "Windows"

    ```bash
    scoop install pandoc
    ```

    ??? note "Manual Pandoc installation"

        ### Manual pandoc installation

        If, for any reason, pandoc installation did not work as expected, you can attempt to install pandoc manually. Follow these instructions to do so and check whether installation was successful.

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

=== "Mac"
    ```bash
    brew install pandoc
    ```

=== "Linux"
    ```bash
    sudo apt install pandoc
    ```

### Installing a python package manager

We recommend installing uv to manage python packages. This can also be used to install deet as a standalone cli app.
=== "Windows"

    In windows, the easiest way to do install uv is using your package manager.
    ```bash
    scoop install uv
    ```

=== "Mac/Linux"

    In Mac/Linux, we recommend you install uv using the [standalone installer](https://docs.astral.sh/uv/getting-started/installation/#__tabbed_1_1)

Now you have the basics set up, continue to [CLI users](#cli-users) if you want to use the CLI, or [package users](#package-users) if you want to use deet as a package, or to [contributors](#contributors) if you want to contribute to `deet`.

## CLI users

If you just want to use the CLI. We recommend you install the package globally into an isolated managed environment. You can do this using `uv tool install <package>`.

To install `deet`

```bash
uv tool install git+https://github.com/destiny-evidence/data-extraction-evaluation-toolkit.git
```

Now you are ready to use `deet`. To test this, run

```bash
deet --help
```

### Installing from a specific branch

If a feature you want to use is being actively worked on, and is not yet merged into main, you can install a specific branch of deet by appending @branch to the previous command. For example

```bash
uv tool install git+https://github.com/destiny-evidence/data-extraction-evaluation-toolkit.git@development
```

will install deet from the development branch.

## Package users

Assuming you are using UV to manage dependencies within your project, you can add deet via

```bash
uv add git+https://github.com/destiny-evidence/data-extraction-evaluation-toolkit.git
uv sync
```

Append @branch to the uv add command to use a specific branch of deet.

## Contributors

If you want to contribute to deet, you will want to clone the repository.

```sh
git clone git@github.com:destiny-evidence/data-extraction-evaluation-toolkit.git # SSH
# or
git clone https://github.com/destiny-evidence/data-extraction-evaluation-toolkit.git # HTTPS
cd data-extraction-evaluation-toolkit
```

Use uv sync to install dependencies (you may need to re-run this if you switch to another branch)

```sh
uv sync
```

### Activate your virtual environment

=== "Windows"

    ```sh
    source .venv/Scripts/activate
    ```

=== "Mac/Linux"

    ```sh
    source .venv/bin/activate
    ```

If you are using an IDE, like VS Code or PyCharm, you can also open the integrated terminal.

### Installing pre-commit hooks

Before committing any changes, install `pre-commit` locally (in your activated `venv`).
This will check any committed changes to aid code consistency.

```sh
pre-commit install
```
