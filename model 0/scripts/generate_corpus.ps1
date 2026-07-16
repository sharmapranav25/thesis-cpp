# PowerShell orchestrator: launches multiple parallel PHP workers,
# each generating a slice of the total target count. All workers share
# the same train_corpus directory.
#
# Usage:
#   .\generate_corpus.ps1 -Total 100000 -Workers 4
#
# Notes:
#   - PHP must be at C:\xampp\php\php.exe (override with -PhpExe).
#   - Worker logs go to <root>/logs/worker_<id>.{out,err}.log.
#   - We use System.Diagnostics.Process directly because Start-Process
#     mis-quotes paths with spaces (we sit under "E:\pranav\model 0\").

[CmdletBinding()]
param(
    [int]    $Total   = 100000,
    [int]    $Workers = 4,
    [string] $OutDir  = 'E:\pranav\model 0\train_corpus',
    [int]    $MinLen  = 4,
    [int]    $MaxLen  = 8,
    [string] $PhpExe  = 'C:\xampp\php\php.exe',
    [string] $Libs    = 'securimage,gregwar,kcaptcha,coolcaptcha,simplecaptcha'
)

$ErrorActionPreference = 'Stop'

$root      = Split-Path -Parent $PSScriptRoot
$generator = Join-Path $root 'generators\gen_corpus.php'
$logs      = Join-Path $root 'logs'
New-Item -ItemType Directory -Force -Path $OutDir, $logs | Out-Null

if (-not (Test-Path $PhpExe)) { throw "php.exe not found at $PhpExe" }
if (-not (Test-Path $generator)) { throw "generator script not found at $generator" }

$base = [int][math]::Floor($Total / $Workers)
$rem  = $Total - ($base * $Workers)
$slices = @()
for ($i = 0; $i -lt $Workers; $i++) {
    $n = $base + ($(if ($i -eq $Workers - 1) { $rem } else { 0 }))
    $slices += $n
}

Write-Output "[orchestrator] launching $Workers workers, total=$Total ($($slices -join ' + '))"
Write-Output "[orchestrator] out=$OutDir min=$MinLen max=$MaxLen libs=$Libs"

# Quote a value for Windows command line (wraps in double quotes if it contains spaces)
function Quote-Arg([string]$v) {
    if ($v -match '\s|"') {
        return '"' + ($v -replace '"','\"') + '"'
    }
    return $v
}

$startedAt = Get-Date
$workerList = New-Object System.Collections.ArrayList
for ($i = 0; $i -lt $Workers; $i++) {
    $count = $slices[$i]
    $seed  = (Get-Random -Minimum 1 -Maximum 2147483647)
    $stdoutPath = Join-Path $logs ("worker_{0:D2}.out.log" -f $i)
    $stderrPath = Join-Path $logs ("worker_{0:D2}.err.log" -f $i)

    $argParts = @(
        (Quote-Arg $generator),
        "--count=$count",
        (Quote-Arg "--out=$OutDir"),
        "--min=$MinLen",
        "--max=$MaxLen",
        "--seed=$seed",
        "--libs=$Libs"
    )
    $argString = $argParts -join ' '

    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $PhpExe
    $psi.Arguments = $argString
    $psi.UseShellExecute = $false
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError  = $true
    $psi.CreateNoWindow = $true

    $p = New-Object System.Diagnostics.Process
    $p.StartInfo = $psi
    [void]$p.Start()

    # Begin async reads to dedicated log files (so output isn't lost when buffers fill)
    $stdoutWriter = [System.IO.StreamWriter]::new($stdoutPath, $false)
    $stderrWriter = [System.IO.StreamWriter]::new($stderrPath, $false)
    $stdoutWriter.AutoFlush = $true
    $stderrWriter.AutoFlush = $true

    Register-ObjectEvent -InputObject $p -EventName OutputDataReceived -MessageData $stdoutWriter -Action {
        if ($EventArgs.Data -ne $null) { $Event.MessageData.WriteLine($EventArgs.Data) }
    } | Out-Null
    Register-ObjectEvent -InputObject $p -EventName ErrorDataReceived -MessageData $stderrWriter -Action {
        if ($EventArgs.Data -ne $null) { $Event.MessageData.WriteLine($EventArgs.Data) }
    } | Out-Null
    $p.BeginOutputReadLine()
    $p.BeginErrorReadLine()

    Write-Output "[orchestrator] worker $i -> $count images, pid=$($p.Id), seed=$seed"
    [void]$workerList.Add([pscustomobject]@{
        Id            = $i
        Count         = $count
        Process       = $p
        StdoutWriter  = $stdoutWriter
        StderrWriter  = $stderrWriter
        StderrPath    = $stderrPath
    })
}

Write-Output "[orchestrator] all workers launched. tail $logs\worker_NN.err.log to watch."

foreach ($w in $workerList) {
    $w.Process.WaitForExit()
    $rc = $w.Process.ExitCode
    Start-Sleep -Milliseconds 250  # let async writers drain
    $w.StdoutWriter.Close()
    $w.StderrWriter.Close()
    Write-Output "[orchestrator] worker $($w.Id) exited rc=$rc"
}

# Clean up event subscriptions
Get-EventSubscriber | Unregister-Event -ErrorAction SilentlyContinue

$elapsed = (Get-Date) - $startedAt
$totalFiles = (Get-ChildItem $OutDir -Recurse -File -Include *.png,*.jpg -ErrorAction SilentlyContinue | Measure-Object).Count
Write-Output "[orchestrator] done. elapsed=$($elapsed.ToString()) files_in_out=$totalFiles"
