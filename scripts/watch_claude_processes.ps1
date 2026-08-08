# Live-refreshing view of processes Claude has launched via the Bash/PowerShell
# tools in this session. Windows can't natively tag "launched by Claude" on a
# process, so this identifies them the honest way: showing each process's real
# command line (via CIM, which plain tasklist doesn't expose) so you can see
# the actual script name (e.g. drc_radio_feature_sweep.py) rather than just
# "python.exe" with no way to tell it apart from anything else on the machine.
# Run this directly in a PowerShell window and leave it open -- it refreshes
# every 3 seconds until you close the window or press Ctrl+C.

while ($true) {
    Clear-Host
    Write-Host "Live process monitor -- processes with a visible script/command line" -ForegroundColor Cyan
    Write-Host "Refreshing every 3s. Close this window or Ctrl+C to stop.`n" -ForegroundColor DarkGray
    Write-Host ("Updated: " + (Get-Date -Format "yyyy-MM-dd HH:mm:ss")) -ForegroundColor DarkGray
    Write-Host ("-" * 100)

    $procs = Get-CimInstance Win32_Process -Filter "Name='python.exe' OR Name='pythonw.exe' OR Name='node.exe' OR Name='chromedriver.exe'" |
        Select-Object ProcessId, Name,
            @{N='MemMB';E={[math]::Round((Get-Process -Id $_.ProcessId -ErrorAction SilentlyContinue).WorkingSet64/1MB,1)}},
            @{N='StartTime';E={try{(Get-Process -Id $_.ProcessId -ErrorAction Stop).StartTime}catch{'?'}}},
            CommandLine

    if (-not $procs) {
        Write-Host "No matching processes currently running." -ForegroundColor Yellow
    } else {
        foreach ($p in $procs) {
            $cl = $p.CommandLine
            if ($cl -and $cl.Length -gt 90) { $cl = $cl.Substring(0,90) + "..." }
            Write-Host ("PID {0,-7} {1,-14} {2,7} MB   started {3}" -f $p.ProcessId, $p.Name, $p.MemMB, $p.StartTime) -ForegroundColor Green
            Write-Host ("    $cl") -ForegroundColor White
        }
    }
    Write-Host ("-" * 100)
    Start-Sleep -Seconds 3
}
