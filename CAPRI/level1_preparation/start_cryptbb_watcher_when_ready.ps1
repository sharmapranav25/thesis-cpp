param(
    [string]$Root = 'E:\pranav\Final sem\Models\level 1',
    [string]$Run = 'cryptbb_capri_v3_phase3_5char_100k_seed42_manual01'
)

$status = Join-Path $Root "PARSeq\logs\$Run\status.json"
$trainerPattern = "*$Run*"

while (-not (Test-Path -LiteralPath $status)) {
    $trainer = Get-CimInstance Win32_Process -Filter "name='python.exe'" |
        Where-Object { $_.CommandLine -like $trainerPattern -and $_.CommandLine -like '*train_cryptbb_v3_100k.py*' }
    if (-not $trainer) { exit 2 }
    Start-Sleep -Seconds 5
}

& (Join-Path $Root '.venv\Scripts\python.exe') `
  (Join-Path $Root 'shared\watch_real_eval.py') `
  $Root `
  $Run `
  (Join-Path $Root 'datasets\cryptbb_capri_v3_phase3_5char_100k\test.csv') `
  --poll-seconds 5
