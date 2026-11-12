param([switch]$Full, [switch]$Install, [string]$PythonExe)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$experimentPython = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
$previousPythonPath = $env:PYTHONPATH
$previousEncoding = $env:PYTHONIOENCODING
try {
    $env:PYTHONPATH = ''
    $env:PYTHONIOENCODING = 'utf-8'
    if (-not (Test-Path -LiteralPath $experimentPython)) {
        if (-not $PythonExe) { throw 'Python 3.12 실행 파일을 -PythonExe로 지정하세요. 기존 환경의 패키지는 변경하지 않습니다.' }
        & $PythonExe -m venv (Join-Path $PSScriptRoot '.venv')
        if ($LASTEXITCODE) { throw '가상환경 생성 실패' }
        $Install = $true
    }
    if ($Install) {
        & $experimentPython -m pip install -r requirements.txt
        if ($LASTEXITCODE) { throw '독립 환경 설치 실패' }
    }
    if (-not (Test-Path -LiteralPath (Join-Path $PSScriptRoot 'checkpoint/model.safetensors'))) {
        & $experimentPython download_model.py
        if ($LASTEXITCODE) { throw '공식 가중치 다운로드 실패' }
    }
    & $experimentPython audit.py
    if ($LASTEXITCODE) { throw '기존 파일 감사 실패' }
    if ($Full) {
        & $experimentPython experiment.py --mode dense --batch 128 --precision fp32 --tag full_evaluation
    } else {
        & $experimentPython experiment.py --mode representative --batch 128 --precision fp32 --runs-per-fault 5
    }
    if ($LASTEXITCODE) { throw '모델 평가 실패: checkpoint를 보존했습니다. 동일 명령으로 재개할 수 있습니다.' }
    if (-not $Full) {
        & $experimentPython finalize.py
        if ($LASTEXITCODE) { throw '보고서/노트북 검증 실패' }
    } else {
        Write-Host '전체 평가 완료: full_evaluation/model_comparison.csv, variable_metrics.csv, baseline_reproduction.json'
    }
} finally {
    $env:PYTHONPATH = $previousPythonPath
    $env:PYTHONIOENCODING = $previousEncoding
}
