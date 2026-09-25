<#
.SYNOPSIS
    Install the Data Extraction Evaluation Toolkit (deet) on Windows.

.DESCRIPTION
    Takes a fresh Windows machine to a working `deet` command. Installs whatever is missing
    (git, uv, pandoc), then installs deet as a uv tool.

    Every step is idempotent: anything already present and usable is left alone.

    Nothing here needs administrator rights, and no terminal restart is needed -- the script
    updates the PATH of the session it runs in as well as the persisted user PATH.

.PARAMETER Ref
    Install from a specific branch or tag instead of the default branch.

.PARAMETER Source
    Install from an arbitrary source (a local path, or another git URL) instead of GitHub.
    Mainly used by CI to install the checked-out revision. Overrides -Ref.

.PARAMETER SkipPandoc
    Do not install pandoc. Pandoc is only needed to work with full-text PDFs; skip it if you
    only use abstracts or pre-processed markdown.

.PARAMETER Force
    Reinstall deet even if it is already installed.

.EXAMPLE
    irm https://raw.githubusercontent.com/destiny-evidence/data-extraction-evaluation-toolkit/main/install.ps1 | iex

.EXAMPLE
    # Install from the development branch. Fetch the script first so it can take arguments.
    & ([scriptblock]::Create((irm https://raw.githubusercontent.com/destiny-evidence/data-extraction-evaluation-toolkit/main/install.ps1))) -Ref development
#>
[CmdletBinding()]
param(
    [string] $Ref,
    [string] $Source,
    [switch] $SkipPandoc,
    [switch] $Force
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# Windows PowerShell 5.1 defaults to TLS 1.0/1.1, which the download hosts reject.
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$RepoUrl = 'https://github.com/destiny-evidence/data-extraction-evaluation-toolkit.git'
$DocsUrl = 'https://destiny-evidence.github.io/data-extraction-evaluation-toolkit/setup/installation/'
$MinUvVersion = [version] '0.9.8'

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

#endregion

#region dependency installation

function Test-ExecutionPolicyAllowsScoop {
    <#
    .SYNOPSIS
        Scoop's installer refuses to run under a Restricted execution policy.
    #>
    $policy = Get-ExecutionPolicy -Scope CurrentUser
    return $policy -notin @('Restricted', 'AllSigned')
}

function Install-Scoop {
    <#
    .SYNOPSIS
        Bootstrap scoop, the package manager recommended in the deet docs.
    .DESCRIPTION
        Scoop installs per-user, so it needs no administrator rights. Run in-process via iex so
        the PATH it sets up is visible to us immediately.
    #>
    if (Test-Command 'scoop') { return }

    if (-not (Test-ExecutionPolicyAllowsScoop)) {
        throw @"
Cannot install scoop: your PowerShell execution policy is '$(Get-ExecutionPolicy -Scope CurrentUser)'.
Run this (it does not need administrator rights), then run the installer again:

    Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
"@
    }

    Write-Info 'Installing scoop (package manager, no admin rights needed)...'
    Invoke-Expression (Invoke-RestMethod -Uri 'https://get.scoop.sh')
    Update-SessionPath
    Add-ToPath (Join-Path $env:USERPROFILE 'scoop\shims')

    if (-not (Test-Command 'scoop')) {
        throw 'scoop was installed but is still not on PATH. Please report this.'
    }
    Add-Report 'Installed scoop'
}

function Install-WithScoop {
    param([Parameter(Mandatory)][string] $Package)

    Install-Scoop
    Write-Info "Installing $Package via scoop..."
    scoop install $Package
    if ($LASTEXITCODE -ne 0) {
        throw "scoop install $Package failed (exit code $LASTEXITCODE)."
    }
    Update-SessionPath
    Add-ToPath (Join-Path $env:USERPROFILE 'scoop\shims')
}

function Initialize-Git {
    <#
    .SYNOPSIS
        Make sure a `git` executable is on PATH.
    .DESCRIPTION
        uv shells out to a real git binary to resolve `git+https://...` dependencies. Without it
        the deet install fails with an error that does not mention git at all.
    #>
    Write-Step 'Checking git'

    if (Test-Command 'git') {
        Write-Good "git found: $((Get-Command git).Source)"
        return
    }

    # Installed, but not on PATH. Very common -- fix it without downloading anything.
    $candidates = @(
        (Join-Path $env:ProgramFiles 'Git\cmd'),
        (Join-Path ${env:ProgramFiles(x86)} 'Git\cmd'),
        (Join-Path $env:LOCALAPPDATA 'Programs\Git\cmd'),
        (Join-Path $env:USERPROFILE 'scoop\shims')
    )
    $found = Find-ExecutableDirectory -Executable 'git.exe' -Candidates $candidates
    if ($found) {
        Write-Info "git is installed at $found but was not on your PATH."
        Add-ToPath $found
        if (Test-Command 'git') {
            Write-Good 'git is now available.'
            Add-Report 'Added the existing git installation to your PATH'
            return
        }
    }

    Write-Info 'git not found, installing...'
    Install-WithScoop -Package 'git'

    if (-not (Test-Command 'git')) {
        throw @"
git could not be installed automatically. Please install it manually, then re-run this script:

    $DocsUrl
"@
    }
    Write-Good 'git installed.'
    Add-Report 'Installed git'
}

function Get-UvVersion {
    $raw = (uv --version) 2>$null
    if (-not $raw) { return $null }
    # e.g. "uv 0.9.8 (abc1234 2025-01-01)"
    if ($raw -match '(\d+)\.(\d+)\.(\d+)') {
        return [version] $Matches[0]
    }
    return $null
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

    $version = Get-UvVersion
    if ($version -and $version -lt $MinUvVersion) {
        Write-Info "uv $version is older than the required $MinUvVersion, updating..."
        uv self update
        Update-SessionPath
        Add-Report "Updated uv to at least $MinUvVersion"
    }
    Write-Good "uv found: $((Get-Command uv).Source) ($(Get-UvVersion))"
}

function Initialize-Pandoc {
    <#
    .SYNOPSIS
        Install pandoc, used for full-text PDF handling.
    .DESCRIPTION
        Non-fatal: deet works on abstracts and pre-processed markdown without it.
    #>
    Write-Step 'Checking pandoc'

    if (Test-Command 'pandoc') {
        Write-Good "pandoc found: $((Get-Command pandoc).Source)"
        return
    }

    $candidates = @(
        (Join-Path $env:ProgramFiles 'Pandoc'),
        (Join-Path $env:LOCALAPPDATA 'Pandoc'),
        (Join-Path $env:USERPROFILE 'scoop\shims')
    )
    $found = Find-ExecutableDirectory -Executable 'pandoc.exe' -Candidates $candidates
    if ($found) {
        Write-Info "pandoc is installed at $found but was not on your PATH."
        Add-ToPath $found
        if (Test-Command 'pandoc') {
            Write-Good 'pandoc is now available.'
            Add-Report 'Added the existing pandoc installation to your PATH'
            return
        }
    }

    try {
        Write-Info 'pandoc not found, installing...'
        Install-WithScoop -Package 'pandoc'
        Write-Good 'pandoc installed.'
        Add-Report 'Installed pandoc'
    }
    catch {
        Write-Warn "Could not install pandoc automatically: $($_.Exception.Message)"
        Write-Warn 'deet will still work on abstracts and markdown, but not on full-text PDFs.'
        Write-Warn "To install pandoc by hand, see: $DocsUrl"
        Add-Report 'Skipped pandoc (installation failed)'
    }
}

function Install-Deet {
    Write-Step 'Installing deet'

    if ($Source) {
        $spec = $Source
    }
    elseif ($Ref) {
        $spec = "git+$RepoUrl@$Ref"
    }
    else {
        $spec = "git+$RepoUrl"
    }

    $uvArgs = @('tool', 'install')
    if ($Force) { $uvArgs += '--force' }
    $uvArgs += $spec

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
    Add-Report "Installed deet from $spec"
}

#endregion

function Invoke-Install {
    Write-Host ''
    Write-Host 'Data Extraction Evaluation Toolkit (deet) installer' -ForegroundColor Cyan

    if (-not $IsWindowsPlatform) {
        throw 'This installer is for Windows. On macOS and Linux, follow the instructions at ' + $DocsUrl
    }

    Initialize-Git
    Initialize-Uv
    if (-not $SkipPandoc) { Initialize-Pandoc } else { Write-Step 'Skipping pandoc (-SkipPandoc)' }
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
    deet --help | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "``deet --help`` failed (exit code $LASTEXITCODE)."
    }
    Write-Good "deet is ready: $((Get-Command deet).Source)"

    Write-Host ''
    Write-Host 'Done.' -ForegroundColor Green
    foreach ($line in $script:Report) { Write-Host "  - $line" }
    Write-Host ''
    Write-Host 'You can use deet now, in this terminal -- no need to restart it.'
    Write-Host 'Change to the folder holding your data and run:'
    Write-Host ''
    Write-Host '    deet --help'
    Write-Host ''
}

# $IsWindows only exists on PowerShell 6+; on Windows PowerShell 5.1 we are on Windows by
# definition. Computed before Invoke-Install so strict mode does not trip on the missing var.
$IsWindowsPlatform = $true
if (Test-Path Variable:\IsWindows) { $IsWindowsPlatform = $IsWindows }

Invoke-Install
