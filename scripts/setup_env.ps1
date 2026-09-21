<#
.SYNOPSIS
    Create the project virtual environment and install all dependencies.

.DESCRIPTION
    PyTorch is installed from the dedicated CPU wheel index first, so that pip
    does not resolve the much larger CUDA build. The development machine for
    this project has no CUDA device; if you do have one, pass -Cuda to install
    the default wheels from PyPI instead.

.EXAMPLE
    .\scripts\setup_env.ps1
    .\scripts\setup_env.ps1 -PythonExe "C:\Python311\python.exe" -Dev
#>
[CmdletBinding()]
param(
    # Interpreter used to create the environment. Python 3.11 or 3.12.
    [string]$PythonExe = "py -3.11",
    # Also install test and linting tooling.
    [switch]$Dev,
    # Install CUDA-enabled PyTorch instead of the CPU-only build.
    [switch]$Cuda
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$venv = Join-Path $root ".venv"
$python = Join-Path $venv "Scripts\python.exe"

if (-not (Test-Path $python)) {
    Write-Host "Creating virtual environment in .venv ..." -ForegroundColor Cyan
    $parts = $PythonExe -split ' '
    & $parts[0] @($parts[1..($parts.Length - 1)]) -m venv $venv
}

Write-Host "Upgrading pip ..." -ForegroundColor Cyan
& $python -m pip install --upgrade pip setuptools wheel --quiet

if ($Cuda) {
    Write-Host "Installing PyTorch (CUDA build from PyPI) ..." -ForegroundColor Cyan
    & $python -m pip install -r requirements-torch-cpu.txt
} else {
    Write-Host "Installing PyTorch (CPU-only wheels) ..." -ForegroundColor Cyan
    & $python -m pip install -r requirements-torch-cpu.txt --index-url https://download.pytorch.org/whl/cpu
}

$reqs = if ($Dev) { "requirements-dev.txt" } else { "requirements.txt" }
Write-Host "Installing $reqs ..." -ForegroundColor Cyan
& $python -m pip install -r $reqs

if (-not (Test-Path (Join-Path $root ".env"))) {
    Copy-Item (Join-Path $root ".env.example") (Join-Path $root ".env")
    Write-Host "Created .env from .env.example" -ForegroundColor Green
}

Write-Host ""
& $python -c "import torch, sys; print(f'python  {sys.version.split()[0]}'); print(f'torch   {torch.__version__}  cuda={torch.cuda.is_available()}')"
Write-Host ""
Write-Host "Done. Activate with:  .\.venv\Scripts\Activate.ps1" -ForegroundColor Green
