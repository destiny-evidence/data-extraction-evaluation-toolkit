<#
.SYNOPSIS
    Install the Data Extraction Evaluation Toolkit (deet) on Windows.

.DESCRIPTION
    Takes a fresh Windows machine to a working `deet` command. Installs uv if it is missing or
    older than deet requires, then installs deet from PyPI as a uv tool.

    Every step is idempotent: anything already present and usable is left alone.

    Nothing here needs administrator rights, and no terminal restart is needed -- the script
    updates the PATH of the session it runs in as well as the persisted user PATH.

.PARAMETER Force
    Reinstall deet even if it is already installed.

.EXAMPLE
    irm https://raw.githubusercontent.com/destiny-evidence/data-extraction-evaluation-toolkit/main/install.ps1 | iex
#>
[CmdletBinding()]
param(
    [switch] $Force
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# Windows PowerShell 5.1 defaults to TLS 1.0/1.1, which the download hosts reject.
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

# deet prints Unicode (an emoji) in its help; a legacy Windows console (cp1252) can't encode it and
# crashes with UnicodeEncodeError. Force UTF-8 stdio, for this session and for future terminals.
$env:PYTHONIOENCODING = 'utf-8'
if (-not [Environment]::GetEnvironmentVariable('PYTHONIOENCODING', 'User')) {
    [Environment]::SetEnvironmentVariable('PYTHONIOENCODING', 'utf-8', 'User')
}

$DocsUrl = 'https://destiny-evidence.github.io/data-extraction-evaluation-toolkit/setup/installation/'
$RawUrl = 'https://raw.githubusercontent.com/destiny-evidence/data-extraction-evaluation-toolkit'

# Summary lines collected as we go, printed at the end.
$script:Report = @()

#region output helpers

function Write-Step {
    param([string] $Message)
    Write-Host ''
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Write-Info {
    param([string] $Message)
    Write-Host "    $Message"
}

function Write-Good {
    param([string] $Message)
    Write-Host "    $Message" -ForegroundColor Green
}

function Write-Warn {
    param([string] $Message)
    Write-Host "    $Message" -ForegroundColor Yellow
}

function Add-Report {
    param([string] $Message)
    $script:Report += $Message
}

#endregion

#region PATH handling
#
# This is the part that makes "no terminal restart" work, and it is easy to get wrong.
#
# Installers write the *persisted* user PATH in the registry. A process that is already running
# -- the terminal you just pasted this into -- keeps its own copy in $env:PATH and never sees
# that write. So after every install we re-read the registry into the live $env:PATH, and when
# we add a directory ourselves we write it to both places.
#
# This only survives because `irm ... | iex` runs the script inside the caller's session. Never
# shell out to a child powershell.exe to do install work: its $env:PATH changes die with it.

function Update-SessionPath {
    <#
    .SYNOPSIS
        Rebuild $env:PATH from the persisted machine and user PATH values.
    .DESCRIPTION
        Directories added to $env:PATH earlier in this run but not persisted anywhere (some
        installers only touch the live session) would be lost by a naive rebuild, so they are
        preserved.
    #>
    $existing = @()
    if ($env:PATH) {
        $existing = $env:PATH -split ';' | Where-Object { $_ }
    }

    $persisted = @()
    foreach ($scope in 'Machine', 'User') {
        $value = [Environment]::GetEnvironmentVariable('PATH', $scope)
        if ($value) {
            $persisted += $value -split ';' | Where-Object { $_ }
        }
    }

    # Persisted entries first (they are the canonical order), then anything session-only.
    $combined = [System.Collections.Generic.List[string]]::new()
    $seen = [System.Collections.Generic.HashSet[string]]::new(
        [System.StringComparer]::OrdinalIgnoreCase)
    foreach ($entry in @($persisted) + @($existing)) {
        $trimmed = $entry.TrimEnd('\')
        if ($trimmed -and $seen.Add($trimmed)) {
            $combined.Add($entry)
        }
    }

    $env:PATH = $combined -join ';'
}

function Add-ToPath {
    <#
    .SYNOPSIS
        Put a directory on the PATH, both for this session and for future ones.
    #>
    param([Parameter(Mandatory)][string] $Directory)

    $normalised = $Directory.TrimEnd('\')

    $userPath = [Environment]::GetEnvironmentVariable('PATH', 'User')
    $userEntries = @()
    if ($userPath) {
        $userEntries = $userPath -split ';' | Where-Object { $_ } | ForEach-Object { $_.TrimEnd('\') }
    }
    if ($userEntries -notcontains $normalised) {
        $updated = (@($userPath) + @($Directory) | Where-Object { $_ }) -join ';'
        [Environment]::SetEnvironmentVariable('PATH', $updated, 'User')
        Write-Info "Added $Directory to your user PATH."
    }

    $sessionEntries = @()
    if ($env:PATH) {
        $sessionEntries = $env:PATH -split ';' | Where-Object { $_ } | ForEach-Object { $_.TrimEnd('\') }
    }
    if ($sessionEntries -notcontains $normalised) {
        $env:PATH = (@($env:PATH) + @($Directory) | Where-Object { $_ }) -join ';'
    }
}

function Test-Command {
    <#
    .SYNOPSIS
        True if a command resolves in the current session.
    .DESCRIPTION
        Application commands are resolved against the live $env:PATH on every call, so re-reading
        $env:PATH earlier in the run is enough for a freshly installed tool to be found here.
    #>
    param([Parameter(Mandatory)][string] $Name)

    return [bool] (Get-Command -Name $Name -CommandType Application -ErrorAction SilentlyContinue)
}

function Find-ExecutableDirectory {
    <#
    .SYNOPSIS
        Look for an executable in well-known install locations.
    .DESCRIPTION
        Covers the common "it is installed, it is just not on PATH" case -- the most frequent
        failure we saw in user testing.
    #>
    param(
        [Parameter(Mandatory)][string] $Executable,
        [Parameter(Mandatory)][string[]] $Candidates
    )

    foreach ($candidate in $Candidates) {
        if (-not $candidate) { continue }
        $full = Join-Path $candidate $Executable
        if (Test-Path -LiteralPath $full -PathType Leaf) {
            return $candidate
        }
    }
    return $null
}

function Test-DirectoryWritable {
    <#
    .SYNOPSIS
        True if a file can be created in the given directory.
    .DESCRIPTION
        deet writes its log into the working directory, so an unwritable one (classically
        C:\WINDOWS\system32, where PowerShell sometimes opens) makes it fail on startup.
    #>
    param([Parameter(Mandatory)][string] $Path)

    try {
        $probe = Join-Path $Path (".deet-write-test-" + [guid]::NewGuid().ToString('N'))
        [System.IO.File]::WriteAllText($probe, '')
        Remove-Item -LiteralPath $probe -Force
        return $true
    }
    catch {
        return $false
    }
}

#endregion

#region dependency installation

function Get-UvVersion {
    $raw = (uv --version) 2>$null
    if (-not $raw) { return $null }
    # e.g. "uv 0.9.8 (abc1234 2025-01-01)"
    if ($raw -match '(\d+)\.(\d+)\.(\d+)') {
        return [version] $Matches[0]
    }
    return $null
}

function Save-DeetPyproject {
    <#
    .SYNOPSIS
        Download the pyproject.toml of the default branch into a new temporary directory.
    #>
    $directory = Join-Path ([System.IO.Path]::GetTempPath()) ('deet-' + [guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory -Path $directory | Out-Null
    Invoke-WebRequest -Uri "$RawUrl/HEAD/pyproject.toml" -OutFile (Join-Path $directory 'pyproject.toml') -UseBasicParsing
    return $directory
}

function Test-UvMeetsRequirement {
    <#
    .SYNOPSIS
        True if the installed uv satisfies the `required-version` in the project's pyproject.toml.
    .DESCRIPTION
        uv enforces `required-version` itself, so this script keeps no version number.
        `uv tool` commands ignore the setting, so a cheap project-level command is used.
    #>
    param([Parameter(Mandatory)][string] $ProjectDirectory)

    # Windows PowerShell 5.1 turns native stderr into a terminating error under 'Stop'.
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        # Stringify each record: formatted ErrorRecords carry a stack-trace style prefix.
        $output = (& uv --project $ProjectDirectory cache dir 2>&1 | ForEach-Object { "$_" }) -join "`n"
        $succeeded = ($LASTEXITCODE -eq 0)
    }
    finally {
        $ErrorActionPreference = $previous
    }
    if (-not $succeeded) { Write-Info $output.Trim() }
    return $succeeded
}

function Update-Uv {
    <#
    .SYNOPSIS
        Update uv with the tool that installed it.
    .DESCRIPTION
        `uv self update` only works for the standalone installer; a uv installed by scoop is updated by scoop.
    #>
    if ((Get-Command uv -CommandType Application).Source -like '*\scoop\*') {
        $update = 'scoop update uv'
        Write-Info "Running: $update"
        scoop update uv
    }
    else {
        $update = 'uv self update'
        Write-Info "Running: $update"
        uv self update
    }
    if ($LASTEXITCODE -ne 0) {
        throw "$update failed (exit code $LASTEXITCODE)."
    }
    Update-SessionPath
}

function Initialize-Uv {
    <#
    .SYNOPSIS
        Make sure uv is on PATH and new enough for deet.
    #>
    Write-Step 'Checking uv'

    if (-not (Test-Command 'uv')) {
        $candidates = @(
            (Join-Path $env:USERPROFILE '.local\bin'),
            (Join-Path $env:USERPROFILE 'scoop\shims')
        )
        $found = Find-ExecutableDirectory -Executable 'uv.exe' -Candidates $candidates
        if ($found) {
            Write-Info "uv is installed at $found but was not on your PATH."
            Add-ToPath $found
        }
    }

    if (-not (Test-Command 'uv')) {
        Write-Info 'uv not found, installing via the official installer...'
        Invoke-Expression (Invoke-RestMethod -Uri 'https://astral.sh/uv/install.ps1')
        Update-SessionPath
        Add-ToPath (Join-Path $env:USERPROFILE '.local\bin')
        Add-Report 'Installed uv'
    }

    if (-not (Test-Command 'uv')) {
        throw 'uv was installed but is still not on PATH. Please report this.'
    }

    $projectDirectory = Save-DeetPyproject
    try {
        if (-not (Test-UvMeetsRequirement $projectDirectory)) {
            Write-Info 'uv is older than deet requires, updating...'
            Update-Uv
            if (-not (Test-UvMeetsRequirement $projectDirectory)) {
                throw "uv still does not meet deet's requirement after updating. Please update uv by hand, then re-run this script: $DocsUrl"
            }
            Add-Report 'Updated uv'
        }
    }
    finally {
        Remove-Item -LiteralPath $projectDirectory -Recurse -Force
    }
    Write-Good "uv found: $((Get-Command uv).Source) ($(Get-UvVersion))"
}

function Install-Deet {
    Write-Step 'Installing deet'

    $uvArgs = @('tool', 'install')
    if ($Force) { $uvArgs += '--force' }
    $uvArgs += 'destiny-deet'

    Write-Info "uv $($uvArgs -join ' ')"
    Write-Info 'This downloads a managed Python and a large set of dependencies; it can take several minutes.'
    & uv @uvArgs
    if ($LASTEXITCODE -ne 0) {
        throw "uv tool install failed (exit code $LASTEXITCODE)."
    }

    # uv installs tool shims into ~\.local\bin and can persist that on PATH for us.
    uv tool update-shell
    Update-SessionPath
    Add-ToPath (Join-Path $env:USERPROFILE '.local\bin')
    Add-Report 'Installed deet'
}

#endregion

function Invoke-Install {
    Write-Host ''
    Write-Host 'Data Extraction Evaluation Toolkit (deet) installer' -ForegroundColor Cyan

    if (-not $IsWindowsPlatform) {
        throw 'This installer is for Windows. On macOS and Linux, follow the instructions at ' + $DocsUrl
    }

    Initialize-Uv
    Install-Deet

    Write-Step 'Verifying'
    if (-not (Test-Command 'deet')) {
        throw @"
deet was installed but the `deet` command is not available in this session.
This should not happen -- please report it, including the output above.

As a workaround for right now:

    `$env:PATH = "`$env:USERPROFILE\.local\bin;`$env:PATH"
"@
    }
    # Run the check from a writable directory: deet creates its log in the working directory, which
    # fails if PowerShell opened somewhere read-only (e.g. C:\WINDOWS\system32).
    Push-Location $HOME
    try {
        deet --help | Out-Null
        if ($LASTEXITCODE -ne 0) {
            throw "``deet --help`` failed (exit code $LASTEXITCODE)."
        }
    }
    finally {
        Pop-Location
    }
    Write-Good "deet is ready: $((Get-Command deet).Source)"

    # If the terminal opened somewhere read-only, move to the home directory so the user's next
    # `deet` command can write its log. Users who started in their own (writable) folder stay put.
    if (-not (Test-DirectoryWritable (Get-Location).Path)) {
        Set-Location $HOME
        Write-Warn "Your terminal opened in a folder you cannot write to; moved you to $HOME."
    }

    Write-Host ''
    Write-Host 'Done.' -ForegroundColor Green
    foreach ($line in $script:Report) { Write-Host "  - $line" }
    Write-Host ''
    Write-Host 'You can now use deet'
    Write-Host 'Change to the folder where you want to create a deet project and run:'
    Write-Host ''
    Write-Host '    deet --help'
    Write-Host ''
}

# $IsWindows only exists on PowerShell 6+; on Windows PowerShell 5.1 we are on Windows by
# definition. Computed before Invoke-Install so strict mode does not trip on the missing var.
$IsWindowsPlatform = $true
if (Test-Path Variable:\IsWindows) { $IsWindowsPlatform = $IsWindows }

Invoke-Install
