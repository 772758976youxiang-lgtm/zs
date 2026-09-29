param([switch]$NoBrowser)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) { throw '请先按 README 安装本地依赖。' }
if (-not (Test-Path -LiteralPath (Join-Path $projectRoot 'frontend\node_modules'))) { throw '请先在 frontend 目录执行 npm ci。' }
$runtimeDir = Join-Path $projectRoot 'workspace'
New-Item -ItemType Directory -Force -Path $runtimeDir | Out-Null
foreach ($servicePort in @(8000,5173)) {
    if (Get-NetTCPConnection -LocalPort $servicePort -State Listen -ErrorAction SilentlyContinue) { throw "端口 $servicePort 已被占用，请使用现有服务，或先运行 scripts/stop_services.py。" }
}
$backendProcess = Start-Process -FilePath $pythonPath -ArgumentList '-m','uvicorn','backend.app.main:app','--host','127.0.0.1','--port','8000' -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $runtimeDir 'backend.log') -RedirectStandardError (Join-Path $runtimeDir 'backend-error.log')
$nodePath = (Get-Command node).Source
$frontendProcess = Start-Process -FilePath $nodePath -ArgumentList 'node_modules/vite/bin/vite.js','--host','127.0.0.1','--port','5173' -WorkingDirectory (Join-Path $projectRoot 'frontend') -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $runtimeDir 'frontend.log') -RedirectStandardError (Join-Path $runtimeDir 'frontend-error.log')
@{ backend=$backendProcess.Id; frontend=$frontendProcess.Id } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $runtimeDir 'services.json') -Encoding utf8
Write-Host 'Unit 工作台：http://127.0.0.1:5173/pretrain'
Write-Host "后端 PID $($backendProcess.Id)，前端 PID $($frontendProcess.Id)。日志保存在 workspace。"
if (-not $NoBrowser) { Start-Process 'http://127.0.0.1:5173/pretrain' }
