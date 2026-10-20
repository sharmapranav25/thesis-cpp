$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$rawRoot = Join-Path $projectRoot 'data\external\outlined_text_captcha\source'
$repository = 'muddokon/outlined-text-captcha'
$revision = '6c6a6a6911ae5b3a288378497cb9139e430dee7b'
$archivePath = Join-Path $rawRoot 'train.zip'
if (Test-Path -LiteralPath $archivePath) { throw "Refusing to overwrite existing archive: $archivePath" }
$tree = Invoke-RestMethod -Uri "https://huggingface.co/api/datasets/$repository/tree/$revision" -TimeoutSec 30
$entry = @($tree | Where-Object { $_.path -eq 'train.zip' })
if ($entry.Count -ne 1 -or $entry[0].size -gt 20000000) { throw 'Archive metadata does not match the bounded download plan.' }
New-Item -ItemType Directory -Path $rawRoot -Force | Out-Null
$downloadUrl = "https://huggingface.co/datasets/$repository/resolve/$revision/train.zip"
Invoke-WebRequest -Uri $downloadUrl -OutFile $archivePath -TimeoutSec 120
$actualSize = (Get-Item -LiteralPath $archivePath).Length
$actualHash = (Get-FileHash -LiteralPath $archivePath -Algorithm SHA256).Hash.ToLowerInvariant()
if ($actualSize -ne $entry[0].size) { throw 'Downloaded archive length differs from pinned repository metadata.' }
if ($entry[0].lfs.oid -and $actualHash -ne $entry[0].lfs.oid) { throw 'Downloaded archive SHA-256 differs from pinned repository metadata.' }
$receipt = [ordered]@{
    repository = $repository
    revision = $revision
    source_page = 'https://huggingface.co/datasets/muddokon/outlined-text-captcha'
    declared_license = 'MIT (Hugging Face dataset metadata; upstream ownership not independently verified)'
    downloaded_at_utc = (Get-Date).ToUniversalTime().ToString('o')
    archive = 'train.zip'
    download_url = $downloadUrl
    bytes = $actualSize
    sha256 = $actualHash
    remote_lfs_sha256 = $entry[0].lfs.oid
    split = 'upstream train only; upstream validation/test archives not downloaded'
    selection = 'Convenience selection from public search results with a declared license and a small downloadable archive; not a statistically random draw from all CAPTCHA datasets.'
    method_settings_changed_for_dataset = $false
    source_code_or_fonts_downloaded = $false
}
$receipt | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $rawRoot 'download_receipt.json') -Encoding UTF8
$receipt | ConvertTo-Json -Depth 5
