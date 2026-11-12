param([switch]$ResultsOnly)
$ErrorActionPreference = 'Stop'
$studyRoot = $PSScriptRoot
$taskRoot = (Resolve-Path -LiteralPath (Join-Path $studyRoot '../..')).Path
Set-Location -LiteralPath $taskRoot
$env:PYTHONPATH = Join-Path $taskRoot '.deps'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONIOENCODING = 'utf-8'
$pythonExe = 'C:\Program Files\WindowsApps\OpenAI.CodexPrimaryRuntime.v26-1104-2876-0_26.1104.2876.0_x64__3k8sg7r9htsxt\dependencies\python\python.exe'
function Invoke-Study([string]$name) {
    & $pythonExe (Join-Path $studyRoot $name)
    if ($LASTEXITCODE) { throw "Failed: $name" }
}
if (-not $ResultsOnly) {
    Invoke-Study 'baseline.py'
    Invoke-Study 'phm_audit.py'
    Invoke-Study 'validate_masks.py'
    & $pythonExe (Join-Path $studyRoot 'degradation.py') --stage pilot
    if ($LASTEXITCODE) { throw 'Pilot failed' }
    & $pythonExe (Join-Path $studyRoot 'degradation.py') --stage full
    if ($LASTEXITCODE) { throw 'Full experiment failed' }
}
Invoke-Study 'finalize.py'
Invoke-Study 'report.py'
Invoke-Study 'notebooks.py'
