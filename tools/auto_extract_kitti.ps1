# auto_extract_kitti.ps1 — Monitors KITTI downloads and auto-extracts when complete
# Run this in a PowerShell window while downloads are in progress:
#   powershell -File "E:\bev_research\tools\auto_extract_kitti.ps1"

$imgZip = "C:\datasets\kitti\data_object_image_2.zip"
$velZip = "C:\datasets\kitti\data_object_velodyne.zip"
$kittiDir = "C:\datasets\kitti"

$imgTarget  = 12597  # MB (expected size)
$velTarget  = 28999  # MB

Write-Host "KITTI Auto-Extractor — monitoring downloads..."
Write-Host "Ctrl+C to cancel"

while ($true) {
    $imgMB = if (Test-Path $imgZip) { [math]::Round((Get-Item $imgZip).Length/1MB,1) } else { 0 }
    $velMB = if (Test-Path $velZip) { [math]::Round((Get-Item $velZip).Length/1MB,1) } else { 0 }

    $imgPct = [math]::Round($imgMB / $imgTarget * 100, 1)
    $velPct = [math]::Round($velMB / $velTarget * 100, 1)

    Write-Host "$(Get-Date -Format 'HH:mm:ss') | Images: $imgMB MB ($imgPct%) | Velodyne: $velMB MB ($velPct%)"

    # Check if images done (within 2% of target)
    if ($imgMB -gt ($imgTarget * 0.98) -and -not (Test-Path "$kittiDir\training\image_2")) {
        Write-Host "[EXTRACTING] Images zip..."
        Expand-Archive -Path $imgZip -DestinationPath $kittiDir -Force
        Write-Host "[DONE] Images extracted."
    }

    # Check if velodyne done
    if ($velMB -gt ($velTarget * 0.98) -and -not (Test-Path "$kittiDir\training\velodyne")) {
        Write-Host "[EXTRACTING] Velodyne zip..."
        Expand-Archive -Path $velZip -DestinationPath $kittiDir -Force
        Write-Host "[DONE] Velodyne extracted."
    }

    # Check if all done
    $imgDone = Test-Path "$kittiDir\training\image_2"
    $velDone = Test-Path "$kittiDir\training\velodyne"
    if ($imgDone -and $velDone) {
        Write-Host "[ALL DONE] KITTI dataset fully downloaded and extracted!"
        Write-Host "Next step: run `conda activate bev_research && python E:\bev_research\tools\verify_data.py`"
        break
    }

    Start-Sleep -Seconds 60
}
