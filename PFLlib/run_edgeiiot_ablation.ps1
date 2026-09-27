# FedAS PA/CS ablation on Edge-IIoT, then a markdown report.
#
# Usage (from PFLlib or the project root):
#   .\run_edgeiiot_ablation.ps1
#   .\run_edgeiiot_ablation.ps1 -ReportOnly
#   .\run_edgeiiot_ablation.ps1 -Force
#
# Result files:
#   PFLlib\results\EdgeIIoT_FedAS_abl_none_0.h5
#   PFLlib\results\EdgeIIoT_FedAS_abl_PAonly_0.h5
#   PFLlib\results\EdgeIIoT_FedAS_abl_CSonly_0.h5
#   PFLlib\results\EdgeIIoT_FedAS_abl_PA_CS_0.h5
#   PFLlib\results\EdgeIIoT_FedAS_abl_report.md

param(
    [int]    $Rounds       = 10,
    [int]    $LocalEpochs  = 5,
    [int]    $BatchSize    = 16,
    [double] $LearningRate = 0.005,
    [int]    $NumClients   = 20,
    [double] $JoinRatio    = 0.6,
    [int]    $NumClasses   = 15,
    [string] $Dataset      = 'EdgeIIoT',
    [string] $Model        = 'MLP',
    [string] $Device       = 'cpu',
    [switch] $ReportOnly,
    [switch] $Force,
    [switch] $WhatIf
)

$ErrorActionPreference = 'Stop'

$pythonCandidates = @(
    'C:\Users\abhil\miniconda3\envs\pfllib\python.exe',
    'python'
)
$python = $pythonCandidates | Where-Object {
    $_ -eq 'python' -or (Test-Path $_)
} | Select-Object -First 1

if (Test-Path (Join-Path $PSScriptRoot 'system\main.py')) {
    $pflRoot  = $PSScriptRoot
    $repoRoot = Split-Path $PSScriptRoot -Parent
} else {
    $repoRoot = $PSScriptRoot
    $pflRoot  = Join-Path $repoRoot 'PFLlib'
}

$systemDir  = Join-Path $pflRoot 'system'
$resultsDir = Join-Path $pflRoot 'results'
$logDir     = Join-Path $repoRoot 'logs'
$reportPy   = Join-Path $systemDir 'utils\write_fedas_abl_report.py'
$configPath = Join-Path $pflRoot "dataset\$Dataset\config.json"
$reportPath = Join-Path $resultsDir "${Dataset}_FedAS_abl_report.md"

$variants = @(
    @{ Goal = 'abl_none';   Extra = @('--no-pa', '--no-cs'); Label = 'Baseline (no PA, no CS)' }
    @{ Goal = 'abl_PAonly'; Extra = @('--no-cs');            Label = 'PA only' }
    @{ Goal = 'abl_CSonly'; Extra = @('--no-pa');            Label = 'CS only' }
    @{ Goal = 'abl_PA_CS';  Extra = @();                     Label = 'Complete FedAS (PA + CS)' }
)

foreach ($d in @($resultsDir, $logDir)) {
    if (-not (Test-Path $d)) { New-Item -ItemType Directory -Path $d | Out-Null }
}

if (-not (Get-Command $python -ErrorAction SilentlyContinue) -and -not (Test-Path $python)) {
    throw "Python not found. Expected the pfllib conda env or python on PATH."
}
if (-not (Test-Path $configPath)) {
    throw "$Dataset client splits are missing. From PFLlib\dataset run: python generate_$Dataset.py noniid - dir"
}
if (-not (Test-Path $reportPy)) {
    throw "Report writer not found: $reportPy"
}

function Write-AblationReport {
    Write-Host "Writing report..." -ForegroundColor Cyan
    Push-Location $systemDir
    try {
        & $python $reportPy `
            --dataset $Dataset `
            --model $Model `
            --results-dir $resultsDir `
            --config $configPath `
            --output $reportPath `
            -ncl $NumClasses `
            -gr $Rounds `
            -ls $LocalEpochs `
            -lbs $BatchSize `
            -lr $LearningRate `
            -nc $NumClients `
            -jr $JoinRatio `
            -dev $Device
        if ($LASTEXITCODE -ne 0) { throw "Report writer failed with exit $LASTEXITCODE" }
    }
    finally {
        Pop-Location
    }
    Write-Host "Report: $reportPath" -ForegroundColor Green
}

if ($ReportOnly) {
    Write-AblationReport
    return
}

$total   = $variants.Count
$index   = 0
$skipped = 0
$failed  = @()
$gridStart = Get-Date

Push-Location $systemDir
try {
    foreach ($variant in $variants) {
        $index++
        $goal  = $variant.Goal
        $h5    = Join-Path $resultsDir "${Dataset}_FedAS_${goal}_0.h5"
        $log   = Join-Path $logDir "${Dataset}_FedAS_${goal}.log"
        $label = "[$index/$total] $($variant.Label)"

        if ((Test-Path $h5) -and -not $Force) {
            Write-Host "$label -- SKIP (result exists)" -ForegroundColor DarkGray
            $skipped++
            continue
        }

        $cliArgs = @(
            'main.py',
            '-data', $Dataset,
            '-ncl',  $NumClasses,
            '-m',    $Model,
            '-algo', 'FedAS',
            '-gr',   $Rounds,
            '-ls',   $LocalEpochs,
            '-lbs',  $BatchSize,
            '-lr',   $LearningRate,
            '-nc',   $NumClients,
            '-jr',   $JoinRatio,
            '-dev',  $Device,
            '-go',   $goal
        ) + $variant.Extra

        if ($WhatIf) {
            Write-Host "$label -- would run:" -ForegroundColor Cyan
            Write-Host "    python $($cliArgs -join ' ')"
            continue
        }

        Write-Host "$label -- running (log: $log)" -ForegroundColor Cyan
        $runStart = Get-Date
        & $python @cliArgs 2>&1 | Tee-Object -FilePath $log
        $exit = $LASTEXITCODE
        $elapsed = (Get-Date) - $runStart

        if ($exit -ne 0 -or -not (Test-Path $h5)) {
            Write-Host ("    FAILED (exit $exit) after {0:hh\:mm\:ss}" -f $elapsed) -ForegroundColor Red
            $failed += $goal
            continue
        }

        Write-Host ("    done in {0:hh\:mm\:ss}" -f $elapsed) -ForegroundColor Green
    }
}
finally {
    Pop-Location
}

$gridElapsed = (Get-Date) - $gridStart
Write-Host ""
Write-Host ("Runs finished in {0:hh\:mm\:ss}. skipped=$skipped failed=$($failed.Count)" -f $gridElapsed) -ForegroundColor Yellow
if ($failed.Count) {
    Write-Host "Failed variants (see logs\):" -ForegroundColor Red
    $failed | ForEach-Object { "  $_" }
}

if (-not $WhatIf) {
    Write-AblationReport
}
