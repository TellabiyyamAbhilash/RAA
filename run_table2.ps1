# Table 2 (Cifar100) grid runner for the FedAS reproduction.
#
# Runs one BETA slice of the grid: all -Algos x all -JoinRatios at the beta that is
# CURRENTLY generated on disk. Beta is baked into the dataset, not a CLI flag, so you
# must regenerate the dataset between beta slices (see FedAS_Experiment_Plan.md sec 3).
#
# Usage (from the project root, "Fed Learning"):
#   .\run_table2.ps1 -Beta 0.1 -Algos FedAvg,FedPer,FedALA,FedAS -JoinRatios 0.6,1.0 -Rounds 10
#
# Resumable: a cell whose .h5 already exists is skipped, so you can Ctrl-C and rerun.

param(
    [Parameter(Mandatory = $true)]
    [string] $Beta,

    [string[]] $Algos = @('FedAvg', 'FedProx', 'FedPer', 'Ditto', 'LG-FedAvg',
                          'FedBABU', 'FedProto', 'FedALA', 'FedPAC', 'FedAS'),

    [double[]] $JoinRatios = @(0.2, 0.4, 0.6, 1.0),

    [int]    $Rounds        = 40,
    [int]    $LocalEpochs   = 5,
    [int]    $BatchSize     = 16,
    [double] $LearningRate  = 0.005,
    [int]    $NumClients    = 20,
    [string] $Dataset       = 'Cifar100',
    [int]    $NumClasses    = 100,
    [string] $Device         = 'cpu',
    [switch] $WhatIf
)

$ErrorActionPreference = 'Stop'

$python     = 'C:\Users\abhil\miniconda3\envs\pfllib\python.exe'
$root       = $PSScriptRoot
$systemDir  = Join-Path $root 'PFLlib\system'
$resultsDir = Join-Path $root 'PFLlib\results'
$logDir     = Join-Path $root 'logs'

foreach ($d in @($resultsDir, $logDir)) {
    if (-not (Test-Path $d)) { New-Item -ItemType Directory -Path $d | Out-Null }
}

if (-not (Test-Path $python)) { throw "pfllib python not found at $python" }

# Sanity-check that the generated dataset matches the beta we were told to run.
$configPath = Join-Path $root "PFLlib\dataset\$Dataset\config.json"
if (-not (Test-Path $configPath)) {
    throw "$Dataset not generated. Run: python generate_$Dataset.py noniid - dir"
}
$config = Get-Content $configPath -Raw | ConvertFrom-Json
if ([string]$config.alpha -ne $Beta) {
    throw ("Dataset on disk has alpha=$($config.alpha) but -Beta $Beta was requested. " +
           "Edit alpha in dataset\utils\dataset_utils.py and regenerate, or pass -Beta $($config.alpha).")
}
Write-Host "Dataset check OK: $Dataset alpha=$($config.alpha), num_clients=$($config.num_clients)" -ForegroundColor Green

$total    = $Algos.Count * $JoinRatios.Count
$index    = 0
$skipped  = 0
$failed   = @()
$timings  = @()
$gridStart = Get-Date

Push-Location $systemDir
try {
    foreach ($algo in $Algos) {
        foreach ($jr in $JoinRatios) {
            $index++
            $goal   = "b${Beta}_P${jr}"
            $h5     = Join-Path $resultsDir "${Dataset}_${algo}_${goal}_0.h5"
            $logTag = "${Dataset}_${algo}_${goal}"
            $log    = Join-Path $logDir "$logTag.log"
            $label  = "[$index/$total] $algo  beta=$Beta  P=$jr"

            if (Test-Path $h5) {
                Write-Host "$label -- SKIP (result exists)" -ForegroundColor DarkGray
                $skipped++
                continue
            }

            $cliArgs = @(
                'main.py',
                '-data', $Dataset,
                '-ncl',  $NumClasses,
                '-m',    'CNN',
                '-algo', $algo,
                '-gr',   $Rounds,
                '-ls',   $LocalEpochs,
                '-lbs',  $BatchSize,
                '-lr',   $LearningRate,
                '-nc',   $NumClients,
                '-jr',   $jr,
                '-dev',  $Device,
                '-go',   $goal
            )

            # FedProx with the default -mu 0.0 is mathematically identical to FedAvg;
            # the paper does not state its mu, so pin one explicitly and record it.
            if ($algo -eq 'FedProx') { $cliArgs += @('-mu', '0.01') }

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
                $failed += $logTag
                continue
            }

            $best = & $python -c @"
import h5py, numpy as np
with h5py.File(r'$h5','r') as f:
    print(round(float(np.array(f['rs_test_acc']).max())*100, 2))
"@
            Write-Host ("    done in {0:hh\:mm\:ss}  best acc = $best %" -f $elapsed) -ForegroundColor Green
            $timings += [pscustomobject]@{
                Algorithm = $algo; Beta = $Beta; P = $jr
                BestAcc   = [double]$best
                Minutes   = [math]::Round($elapsed.TotalMinutes, 1)
                PerRound  = [math]::Round($elapsed.TotalMinutes / ($Rounds + 1), 2)
            }
        }
    }
}
finally {
    Pop-Location
}

$gridElapsed = (Get-Date) - $gridStart
Write-Host ""
Write-Host ("Grid finished in {0:hh\:mm\:ss}. ran=$($timings.Count) skipped=$skipped failed=$($failed.Count)" -f $gridElapsed) -ForegroundColor Yellow
if ($timings.Count) { $timings | Sort-Object Algorithm, P | Format-Table -AutoSize }
if ($failed.Count)  { Write-Host "Failed cells (see logs\):" -ForegroundColor Red; $failed | ForEach-Object { "  $_" } }
