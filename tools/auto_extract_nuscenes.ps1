# auto_extract_nuscenes.ps1
# Monitors all nuScenes blob downloads and auto-extracts each when complete
# Run in a separate PowerShell terminal:
#   powershell -File "E:\bev_research\tools\auto_extract_nuscenes.ps1"

$destDir = "C:\datasets\nuscenes"
$expectedGB = @{  # Expected sizes in GB (approximate)
    "v1.0-trainval01_blobs.tgz" = 29
    "v1.0-trainval02_blobs.tgz" = 29
    "v1.0-trainval03_blobs.tgz" = 29
    "v1.0-trainval04_blobs.tgz" = 29
    "v1.0-trainval05_blobs.tgz" = 29
    "v1.0-trainval06_blobs.tgz" = 29
    "v1.0-trainval07_blobs.tgz" = 29
    "v1.0-trainval08_blobs.tgz" = 29
    "v1.0-trainval09_blobs.tgz" = 29
    "v1.0-trainval10_blobs.tgz" = 29
}

$extracted = @{}
$total = $expectedGB.Count

Write-Host "nuScenes Auto-Monitor — watching $total blob files + metadata"
Write-Host "Each blob is ~29 GB. Total: ~300 GB"
Write-Host "Press Ctrl+C to stop monitoring (downloads continue in background)"

while ($true) {
    $done = 0
    $totalMB = 0
    foreach ($file in $expectedGB.Keys) {
        $path = "$destDir\$file"
        if (Test-Path $path) {
            $mb = [math]::Round((Get-Item $path).Length / 1MB, 1)
            $targetMB = $expectedGB[$file] * 1000
            $pct = [math]::Min(100, [math]::Round($mb / $targetMB * 100, 1))
            $totalMB += $mb
            if ($pct -ge 98 -and -not $extracted[$file]) {
                # Extract this blob
                Write-Host "[$(Get-Date -Format 'HH:mm')] EXTRACTING $file..."
                tar -xzf $path -C $destDir
                $extracted[$file] = $true
                Write-Host "[$(Get-Date -Format 'HH:mm')] DONE extracting $file"
                $done++
            }
            elseif ($extracted[$file]) {
                $done++
            }
        }
    }
    
    $totalGB = [math]::Round($totalMB / 1024, 1)
    Write-Host "$(Get-Date -Format 'HH:mm:ss') | Blobs complete: $done/$total | Total downloaded: $totalGB GB / ~290 GB"
    Get-Job | Where-Object { $_.State -eq 'Running' } | Measure-Object | ForEach-Object { Write-Host "  Active download jobs: $($_.Count)" }
    
    if ($done -eq $total) {
        Write-Host "`n[ALL DONE] All nuScenes blobs downloaded and extracted!"
        Write-Host "Now extract metadata:"
        Write-Host "  tar -xzf $destDir\v1.0-trainval_meta.tgz -C $destDir"
        Write-Host "Then run verification:"
        Write-Host "  conda activate bev_research && python E:\bev_research\tools\verify_data.py"
        break
    }
    Start-Sleep -Seconds 120  # Check every 2 minutes
}
