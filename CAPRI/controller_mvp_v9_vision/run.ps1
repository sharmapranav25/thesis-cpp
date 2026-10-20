param(
    [ValidateSet('sam','dinov3')][string]$Backend = 'sam',
    [string]$Checkpoint = ''
)
$ErrorActionPreference = 'Stop'
$taskPython = 'E:/pranav/Final sem/sourcefree_synth/experiments_xgb/.venv/Scripts/python.exe'
$taskArguments = @((Join-Path $PSScriptRoot 'build9.py'), '--backend', $Backend)
if ($Checkpoint) { $taskArguments += @('--checkpoint', $Checkpoint) }
& $taskPython @taskArguments
if ($LASTEXITCODE -ne 0) { throw 'Controller build failed; see output above.' }
$taskGateArguments = @((Join-Path $PSScriptRoot 'quality_gate9.py'), '--backend', $Backend)
if ($Checkpoint) { $taskGateArguments += @('--checkpoint', $Checkpoint) }
& $taskPython @taskGateArguments
if ($LASTEXITCODE -ne 0) { throw 'Controller quality gate failed; see output above.' }
& $taskPython (Join-Path $PSScriptRoot 'review9.py') --backend $Backend
if ($LASTEXITCODE -ne 0) { throw 'Controller review failed; see output above.' }
