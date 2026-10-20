$root = 'E:\pranav\Final sem\Models\level 1'
$run = 'cryptbb_capri_v3_phase3_5char_100k_seed42_manual02'
$completed = Join-Path $root "PARSeq\logs\$run\completed.json"
$source = Join-Path $root "PARSeq\checkpoints\$run\last.pt"
$destinationDirectory = Join-Path $root "PARSeq\checkpoints\$run\per_epoch"
$destination = Join-Path $destinationDirectory 'epoch_15.pt'

while (-not (Test-Path -LiteralPath $completed)) {
    $trainer = Get-CimInstance Win32_Process -Filter "name='python.exe'" |
        Where-Object { $_.CommandLine -like "*$run*" -and $_.CommandLine -like '*train_cryptbb_v3_100k.py*' }
    if (-not $trainer) { exit 2 }
    Start-Sleep -Seconds 10
}

if (-not (Test-Path -LiteralPath $destination)) {
    New-Item -ItemType Directory -Path $destinationDirectory -Force | Out-Null
    Copy-Item -LiteralPath $source -Destination $destination
}
