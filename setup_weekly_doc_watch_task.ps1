$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$script = Join-Path $root "run_weekly_doc_watch.ps1"
$taskName = "T2S GPT Weekly Doc Watch"

if (-not (Test-Path -LiteralPath $script)) {
    throw "Missing watcher script: $script"
}

$action = New-ScheduledTaskAction `
    -Execute "powershell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$script`"" `
    -WorkingDirectory $root

$trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Sunday -At "03:30AM"

$settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit (New-TimeSpan -Hours 3)

try {
    $principal = New-ScheduledTaskPrincipal `
        -UserId $env:USERNAME `
        -LogonType S4U `
        -RunLevel Limited

    Register-ScheduledTask `
        -TaskName $taskName `
        -Action $action `
        -Trigger $trigger `
        -Settings $settings `
        -Principal $principal `
        -Force | Out-Null
} catch {
    $quotedScript = '"' + $script + '"'
    $taskRun = "powershell.exe -NoProfile -ExecutionPolicy Bypass -File $quotedScript"
    & schtasks.exe /Create /TN $taskName /TR $taskRun /SC WEEKLY /D SUN /ST 03:30 /F | Out-Host
    if ($LASTEXITCODE -ne 0) {
        throw
    }
}

Write-Host "Registered: $taskName"
Write-Host "Runs weekly on Sunday at 03:30."
Write-Host "Run now: Start-ScheduledTask -TaskName '$taskName'"
