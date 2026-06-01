$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

$env:PYTHONUTF8 = "1"
$logDir = Join-Path $root "output\t2s_doc_watch"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$logPath = Join-Path $logDir "scheduled-$stamp.log"
$latestPath = Join-Path $logDir "scheduled-latest.log"

python .\t2s_doc_watcher.py 2>&1 | Tee-Object -FilePath $logPath
$exitCode = $LASTEXITCODE
Copy-Item -LiteralPath $logPath -Destination $latestPath -Force
exit $exitCode

