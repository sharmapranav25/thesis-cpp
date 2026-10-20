param([Parameter(Mandatory=$true)][ValidateSet('Switch','Resume','Cleanup')][string]$Action)
$ErrorActionPreference = 'Stop'
$capriWorkspace = 'C:\Users\pranav\Documents\ChatGPT\thesis'
$capriRuns = Join-Path $capriWorkspace 'runs'
$capriSource = Join-Path $capriRuns 'capri_decoder_500k_20261007'
$capriBackup = Join-Path $capriRuns 'capri_decoder_500k_20261007.verified_c_backup'
$capriDestination = 'E:\pranav\CAPRI\runs\capri_decoder_500k_20261007'
$capriJournal = Join-Path $capriRuns 'capri_storage_migration_20261007'
$capriReady = Get-Content -LiteralPath (Join-Path $capriJournal 'cutover_ready.json') -Raw | ConvertFrom-Json
if (-not $capriReady.verified -or $capriReady.source -ne $capriSource -or $capriReady.destination -ne $capriDestination) { throw 'Unverified migration roots' }
$capriHasher = [System.Security.Cryptography.SHA256]::Create()
$capriManifestStream = [System.IO.File]::OpenRead((Join-Path $capriJournal 'verified_manifest.json'))
try { $capriManifestHash = [BitConverter]::ToString($capriHasher.ComputeHash($capriManifestStream)).Replace('-','').ToLowerInvariant() }
finally { $capriManifestStream.Dispose(); $capriHasher.Dispose() }
if ($capriManifestHash -ne $capriReady.manifest_sha256) { throw 'Migration manifest checksum mismatch' }
if ((Resolve-Path -LiteralPath $capriDestination).Path -ne $capriDestination) { throw 'Unexpected destination resolution' }
if ($Action -eq 'Switch') {
    if ((Resolve-Path -LiteralPath $capriSource).Path -ne $capriSource) { throw 'Unexpected source resolution' }
    if ((Get-Item -LiteralPath $capriSource).Attributes -band [IO.FileAttributes]::ReparsePoint) { throw 'Source already linked' }
    if ([IO.Path]::GetFullPath($capriBackup) -ne $capriBackup -or [IO.Path]::GetDirectoryName($capriBackup) -ne $capriRuns) { throw 'Backup escapes intended runs folder' }
    if (Test-Path -LiteralPath $capriBackup) { throw 'Preserve existing backup' }
    Copy-Item -LiteralPath (Join-Path $capriWorkspace 'configs/local.json') -Destination (Join-Path $capriJournal 'local.before.json')
    Rename-Item -LiteralPath $capriSource -NewName ([IO.Path]::GetFileName($capriBackup))
    try {
        New-Item -ItemType Junction -Path $capriSource -Target $capriDestination | Out-Null
        $capriConfig = Get-Content -LiteralPath (Join-Path $capriWorkspace 'configs/local.json') -Raw | ConvertFrom-Json
        $capriConfig | Add-Member -NotePropertyName decoder_storage -NotePropertyValue ([ordered]@{logical_path='runs/capri_decoder_500k_20261007';physical_path=$capriDestination}) -Force
        $capriConfig | ConvertTo-Json -Depth 30 | Set-Content -LiteralPath (Join-Path $capriWorkspace 'configs/local.json') -Encoding UTF8
    } catch {
        throw 'Switch incomplete. Preserve both copies and inspect the journal before recovery.'
    }
} elseif ($Action -eq 'Resume') {
    $capriLink = Get-Item -LiteralPath $capriSource
    if (-not ($capriLink.Attributes -band [IO.FileAttributes]::ReparsePoint) -or $capriLink.Target -ne $capriDestination) { throw 'Incorrect directory link' }
    $capriStamp = Get-Date -Format 'yyyyMMdd_HHmmss'
    $capriOut = Join-Path $capriDestination "baseline_localization.after_E_$capriStamp.stdout.log"
    $capriErr = Join-Path $capriDestination "baseline_localization.after_E_$capriStamp.stderr.log"
    $capriProcess = Start-Process -FilePath 'E:/pranav/Final sem/sourcefree_synth/experiments_xgb/.venv/Scripts/python.exe' -ArgumentList '-B','capri.py','decoder','train-baseline','--workers','4' -WorkingDirectory $capriWorkspace -RedirectStandardOutput $capriOut -RedirectStandardError $capriErr -WindowStyle Hidden -PassThru
    @{launcher_pid=$capriProcess.Id;stdout=$capriOut;stderr=$capriErr} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $capriJournal 'restart.json') -Encoding UTF8
} else {
    if (-not (Test-Path -LiteralPath (Join-Path $capriJournal 'resume_verified.json'))) { throw 'Resume must be verified before cleanup' }
    $capriVerified = Get-Content -LiteralPath (Join-Path $capriJournal 'resume_verified.json') -Raw | ConvertFrom-Json
    if ($capriVerified.manifest_sha256 -ne $capriManifestHash -or $capriVerified.physical_root -ne $capriDestination) { throw 'Resume verification mismatch' }
    $capriResolvedBackup = (Resolve-Path -LiteralPath $capriBackup).Path
    if ($capriResolvedBackup -ne $capriBackup -or [IO.Path]::GetDirectoryName($capriResolvedBackup) -ne $capriRuns) { throw 'Cleanup target escapes intended backup' }
    if ((Get-Item -LiteralPath $capriBackup).Attributes -band [IO.FileAttributes]::ReparsePoint) { throw 'Never recursively remove a reparse point' }
    $capriLink = Get-Item -LiteralPath $capriSource
    if (-not ($capriLink.Attributes -band [IO.FileAttributes]::ReparsePoint) -or $capriLink.Target -ne $capriDestination) { throw 'Live directory link is incorrect' }
    Remove-Item -LiteralPath $capriResolvedBackup -Recurse -Force
}
