# One-time setup so the dashboard can be opened from your phone over Tailscale
# (a private network between your own devices - nothing is exposed to the internet).
#
#   Double-click setup_tailscale.bat (it asks for admin once, for the installer).
#
# What it does: installs Tailscale, waits for you to sign in once in the window
# that opens, then tells Tailscale to serve the dashboard (127.0.0.1:5000) to
# your other devices only, and prints the address to open on the phone.
$ErrorActionPreference = "Stop"
$ts = "C:\Program Files\Tailscale\tailscale.exe"

if (-not (Test-Path $ts)) {
    Write-Host "Installing Tailscale (winget, ~30 MB from pkgs.tailscale.com)..."
    winget install --id Tailscale.Tailscale -e --accept-package-agreements --accept-source-agreements
    if (-not (Test-Path $ts)) { throw "Tailscale did not install - open https://tailscale.com/download/windows and install it by hand, then run this again." }
}

Write-Host ""
Write-Host "Signing in: a browser window opens - use the SAME account you will use on the phone (Google / Apple / Microsoft / GitHub)."
try { & $ts up --accept-routes=false 2>&1 | ForEach-Object { Write-Host $_ } } catch { }

$deadline = (Get-Date).AddMinutes(10)
$name = $null
while ((Get-Date) -lt $deadline) {
    try {
        $st = & $ts status --json 2>$null | ConvertFrom-Json
        if ($st -and $st.BackendState -eq "Running") { $name = $st.Self.DNSName.TrimEnd("."); break }
    } catch { }
    Start-Sleep -Seconds 3
}
if (-not $name) { throw "Not signed in yet. Run this script again after signing in from the Tailscale tray icon." }

# Serve the local dashboard to the tailnet only (plain HTTP on port 80 of this machine's tailnet name).
& $ts serve --bg --http=80 http://127.0.0.1:5000 | Out-Null
$ip = (& $ts ip -4).Trim()
Write-Host ""
Write-Host "Done. On the phone (Tailscale app signed in to the same account, VPN toggle on) open:"
Write-Host "    http://$($name.Split('.')[0])      or      http://$ip"
Write-Host "The laptop keeps serving it after a reboot (Tailscale runs as a service)."
