# data-extraction-evaluation-toolkit

A suite of tools, data models, etc. for extracting data from documents (e.g. papers) and evaluating the performance of such extraction tasks.

## PANDOC installation and verification on Windows

### 1. Installation

There are multiple options. The first and possibly best one is using an installer file provided via GitHub. For the most recent release, visit:

- https://github.com/jgm/pandoc/releases

Select the `.msi` file from the latest release (e.g. `pandoc-3.8.3-windows-x86_64.msi` for release 3.8.3, as of January 2026) and run the installer by double-clicking on it.

Alternatively, one may download the ZIP file from the link above, unpack it, and move the binaries to a directory of your choice (not tested; more information is available via the link below).

For other options (Chocolatey or Winget package manager, or Conda Forge), see:

- https://pandoc.org/installing.html

**Note:** Using the `.msi` method should remove older versions of Pandoc and update PATH variables automatically, while other methods may cause multiple versions to be installed.

---

### 2. Verifying that Pandoc works

These instructions are taken from:
- https://pandoc.org/getting-started.html

#### Steps

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

### 3. Completion

   If the HTML file opens and displays well then you have successfully installed Pandoc and converted a Markdown file to HTML.
