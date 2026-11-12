param([switch]$SkipAudit)
$ErrorActionPreference = 'Stop'
$taskRoot = $PSScriptRoot
Set-Location -LiteralPath $taskRoot
$env:PYTHONPATH = Join-Path $taskRoot '.deps'
$pythonExe = 'C:\Program Files\WindowsApps\OpenAI.CodexPrimaryRuntime.v26-1104-2876-0_26.1104.2876.0_x64__3k8sg7r9htsxt\dependencies\python\python.exe'
if (-not $SkipAudit) {
    & $pythonExe src/data/audit.py
    if ($LASTEXITCODE) { throw 'Dataset A audit failed' }
    foreach ($rdataFile in Get-ChildItem -LiteralPath $taskRoot -Filter '*.RData') {
        & $pythonExe src/data/audit.py $rdataFile.Name
        if ($LASTEXITCODE) { throw "Dataset B audit failed: $($rdataFile.Name)" }
    }
}
& $pythonExe tools/build_phase1.py
if ($LASTEXITCODE) { throw 'Report generation failed' }
& $pythonExe -m unittest discover -s tests
if ($LASTEXITCODE) { throw 'Validation failed' }
& $pythonExe tools/execute_notebooks.py
if ($LASTEXITCODE) { throw 'Notebook execution failed' }
