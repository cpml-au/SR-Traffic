# Prints the YAML frontmatter of every phase file in assets/phases/.
# Run from the automodel/ directory:
#   pwsh scripts/read_phases.ps1

$phasesDir = Join-Path $PSScriptRoot "..\assets\phases"

foreach ($file in Get-ChildItem -Path $phasesDir -Filter "*.md") {
    Write-Host "=== $($file.Name) ==="
    $lines = Get-Content $file.FullName
    foreach ($line in $lines[1..($lines.Count - 1)]) {
        if ($line -eq "---") { break }
        Write-Host $line
    }
    Write-Host ""
}
