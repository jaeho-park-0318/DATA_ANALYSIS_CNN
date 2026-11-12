param([switch]$ReportsOnly)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$env:PYTHONPATH = Join-Path $PSScriptRoot '.deps'
$env:MPLCONFIGDIR = Join-Path $PSScriptRoot '.matplotlib'
$pythonExe = 'C:\Program Files\WindowsApps\OpenAI.CodexPrimaryRuntime.v26-1104-2876-0_26.1104.2876.0_x64__3k8sg7r9htsxt\dependencies\python\python.exe'
function Invoke-Phase2Script([string]$scriptPath) {
    & $pythonExe $scriptPath
    if ($LASTEXITCODE) { throw "Failed: $scriptPath" }
}
if (-not $ReportsOnly) {
    Invoke-Phase2Script 'src/data/phase2.py'
    Invoke-Phase2Script 'src/features/phase2.py'
    Invoke-Phase2Script 'src/models/phase2.py'
    Invoke-Phase2Script 'src/models/temporal.py'
    Invoke-Phase2Script 'src/evaluation/run_phase2.py'
    Invoke-Phase2Script 'tools/phase2_explain.py'
    Invoke-Phase2Script 'tools/package_phase2.py'
    Invoke-Phase2Script 'tools/validate_phase2.py'
}
Invoke-Phase2Script 'tools/build_phase2_report.py'
& $pythonExe -m unittest discover -s tests -v 2>&1 | Tee-Object -FilePath reports/sources/phase2/test_validation.txt
if ($LASTEXITCODE) { throw 'Tests failed' }
Invoke-Phase2Script 'tools/execute_notebooks.py'
