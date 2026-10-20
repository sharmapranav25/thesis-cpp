$ErrorActionPreference = 'Stop'
$bankRoot = Split-Path -Parent $PSScriptRoot
$resourceRoot = Join-Path $bankRoot 'resources/feature_bank_01'
if (Test-Path -LiteralPath $resourceRoot) { throw 'Refusing to overwrite resource acquisition' }
New-Item -ItemType Directory -Path $resourceRoot | Out-Null
$receipt = @()
foreach ($repo in @('Gregwar/Captcha','dapphp/securimage','mobicms/captcha','yasirmturk/simple-php-captcha','laminas/laminas-captcha')) {
    $name = $repo.Replace('/','_')
    $commit = Invoke-RestMethod -Uri "https://api.github.com/repos/$repo/commits/HEAD" -Headers @{'User-Agent'='feature-bank-research'}
    $revision = $commit.sha
    $url = "https://codeload.github.com/$repo/zip/$revision"
    $archive = Join-Path $resourceRoot "$name.zip"
    Invoke-WebRequest -Uri $url -OutFile $archive
    Expand-Archive -LiteralPath $archive -DestinationPath (Join-Path $resourceRoot $name)
    $receipt += @{repository=$repo;revision=$revision;url=$url;sha256=(Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLower();withheld=($repo -eq 'laminas/laminas-captcha')}
    $receipt | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $resourceRoot 'acquisition.json') -Encoding utf8
}
$phpUrl = 'https://downloads.php.net/~windows/releases/archives/php-8.5.11-nts-Win32-vs17-x64.zip'
$phpZip = Join-Path $resourceRoot 'php-8.5.11.zip'
Invoke-WebRequest -Uri $phpUrl -OutFile $phpZip
$phpHash = (Get-FileHash -LiteralPath $phpZip -Algorithm SHA256).Hash.ToLower()
if ($phpHash -ne '0ea96e0d2b9b737a6036f05cf4e95c49313faa6d0f27bd97edb2742503f0c043') { throw 'PHP publisher checksum mismatch' }
Expand-Archive -LiteralPath $phpZip -DestinationPath (Join-Path $resourceRoot 'php')
@{url=$phpUrl;sha256=$phpHash;source='https://www.php.net/downloads.php?os=windows';isolated=$true} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $resourceRoot 'php_receipt.json') -Encoding utf8
& (Join-Path $resourceRoot 'php/php.exe') -n -d "extension_dir=$resourceRoot/php/ext" -d extension=gd -v
if ($LASTEXITCODE -ne 0) { throw 'PHP runtime preflight failed' }
