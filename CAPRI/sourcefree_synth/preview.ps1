param(
    [Parameter(Mandatory=$true)][string]$RealDir,
    [string[]]$ExcludeDir = @(),
    [ValidateSet('no-vlm', 'light-vlm')][string]$Mode = 'no-vlm',
    [ValidateSet('pixels', 'dinov2')][string]$Features = 'pixels',
    [string]$Python = 'E:\pranav\model 1\.venv\Scripts\python.exe',
    [string]$Lengths = '4,5,6,7',
    [string]$Charset = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
    [int]$Trials = 12,
    [int]$Samples = 16,
    [int]$Count = 64,
    [string]$FontDir = '',
    [string]$VlmModel = 'Qwen/Qwen2-VL-2B-Instruct',
    [string]$CacheDir = 'E:\hf_cache\hub'
)
$ErrorActionPreference = 'Stop'
if (-not (Test-Path -LiteralPath $Python)) { throw "Python not found: $Python. Supply -Python with your interpreter path." }
$runName = '{0}_{1}_{2}' -f $Mode, (Get-Date -Format 'yyyyMMdd_HHmmss'), ([guid]::NewGuid().ToString('N').Substring(0, 6))
$fitOutput = Join-Path $PSScriptRoot ('runs\' + $runName + '_fit')
$previewOutput = Join-Path $PSScriptRoot ('runs\' + $runName + '_preview')
$fitArgs = @('-B', '-m', 'sourcefree_synth', 'fit', '--real-dir', $RealDir, '--out', $fitOutput,
    '--mode', $Mode, '--features', $Features, '--lengths', $Lengths, '--charset', $Charset,
    '--trials', $Trials, '--samples', $Samples, '--vlm-model', $VlmModel, '--cache-dir', $CacheDir)
foreach ($directory in $ExcludeDir) { $fitArgs += @('--exclude-dir', $directory) }
if ($FontDir) { $fitArgs += @('--font-dir', $FontDir) }
Push-Location $PSScriptRoot
try {
    & $Python @fitArgs
    if ($LASTEXITCODE -ne 0) { throw "Fit failed; inspect $fitOutput\status.json and trials.jsonl" }
    & $Python -B -m sourcefree_synth generate --config (Join-Path $fitOutput 'generator.json') --out $previewOutput --count $Count
    if ($LASTEXITCODE -ne 0) { throw "Generation failed; inspect $previewOutput\status.json" }
    & $Python -B -m sourcefree_synth audit --run $previewOutput
    if ($LASTEXITCODE -ne 0) { throw 'Generated dataset audit failed' }
    Write-Host "Real validation panel: $fitOutput\real_validation.png"
    Write-Host "Synthetic preview: $previewOutput\preview.png"
    Write-Host "Style report: $fitOutput\report.json"
} finally {
    Pop-Location
}
