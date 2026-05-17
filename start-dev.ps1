param(
    [string]$ConfigPath = (Join-Path $PSScriptRoot 'dev-launch.config.json'),
    [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Get-ConfigValue {
    param(
        [Parameter(Mandatory = $false)]$Object,
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $false)]$Default = $null
    )

    if ($null -eq $Object) {
        return $Default
    }

    if ($Object.PSObject.Properties.Name -contains $Name) {
        return $Object.$Name
    }

    return $Default
}

function Resolve-WorkspacePath {
    param(
        [Parameter(Mandatory = $true)][string]$RootPath,
        [Parameter(Mandatory = $false)][string]$Candidate
    )

    if ([string]::IsNullOrWhiteSpace($Candidate)) {
        return ''
    }

    if ([System.IO.Path]::IsPathRooted($Candidate)) {
        return [System.IO.Path]::GetFullPath($Candidate)
    }

    return [System.IO.Path]::GetFullPath((Join-Path $RootPath $Candidate))
}

function Start-CommandWindow {
    param(
        [Parameter(Mandatory = $true)][string]$Title,
        [Parameter(Mandatory = $true)][string]$Command,
        [Parameter(Mandatory = $true)][bool]$IsDryRun
    )

    Write-Host "[$Title]" -ForegroundColor Cyan
    Write-Host $Command

    if ($IsDryRun) {
        return
    }

    Start-Process -FilePath 'powershell.exe' -ArgumentList @(
        '-NoExit',
        '-ExecutionPolicy', 'Bypass',
        '-Command', $Command
    ) | Out-Null
}

$rootPath = [System.IO.Path]::GetFullPath($PSScriptRoot)
$configFullPath = Resolve-WorkspacePath -RootPath $rootPath -Candidate $ConfigPath

if (-not (Test-Path -LiteralPath $configFullPath)) {
    throw "Config file not found: $configFullPath"
}

$venvPython = Join-Path $rootPath '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $venvPython)) {
    throw "Python virtual environment not found: $venvPython"
}

$npmCommand = (Get-Command npm.cmd -ErrorAction Stop).Source
$config = Get-Content -LiteralPath $configFullPath -Raw | ConvertFrom-Json

$backendConfig = Get-ConfigValue -Object $config -Name 'backend' -Default $null
$frontendConfig = Get-ConfigValue -Object $config -Name 'frontend' -Default $null
$sam3Config = Get-ConfigValue -Object $config -Name 'sam3' -Default $null

$backendEnabled = [bool](Get-ConfigValue -Object $backendConfig -Name 'enabled' -Default $true)
$backendHost = [string](Get-ConfigValue -Object $backendConfig -Name 'host' -Default '127.0.0.1')
$backendPort = [int](Get-ConfigValue -Object $backendConfig -Name 'port' -Default 8000)
$backendMysqlUrl = [string](Get-ConfigValue -Object $backendConfig -Name 'mysqlUrl' -Default '')
$backendDbBackend = [string](Get-ConfigValue -Object $backendConfig -Name 'dbBackend' -Default 'auto')

$frontendEnabled = [bool](Get-ConfigValue -Object $frontendConfig -Name 'enabled' -Default $true)
$frontendHost = [string](Get-ConfigValue -Object $frontendConfig -Name 'host' -Default '127.0.0.1')
$frontendPort = [int](Get-ConfigValue -Object $frontendConfig -Name 'port' -Default 5173)
$defaultProxyBackendTarget = 'http://{0}:{1}' -f $backendHost, $backendPort
$proxyBackendTarget = [string](Get-ConfigValue -Object $frontendConfig -Name 'proxyBackendTarget' -Default $defaultProxyBackendTarget)

$sam3Enabled = [bool](Get-ConfigValue -Object $sam3Config -Name 'enabled' -Default $false)
$sam3RunImportCheck = [bool](Get-ConfigValue -Object $sam3Config -Name 'runImportCheck' -Default $false)
$sam3Device = [string](Get-ConfigValue -Object $sam3Config -Name 'device' -Default 'cuda')
$sam3CheckpointPath = Resolve-WorkspacePath -RootPath $rootPath -Candidate ([string](Get-ConfigValue -Object $sam3Config -Name 'checkpointPath' -Default 'MedicalSAM3/checkpoint/MedSAM3.pt'))
$sam3LoraEnabled = [bool](Get-ConfigValue -Object $sam3Config -Name 'loraEnabled' -Default $false)
$sam3LoraPath = Resolve-WorkspacePath -RootPath $rootPath -Candidate ([string](Get-ConfigValue -Object $sam3Config -Name 'loraPath' -Default ''))
$sam3LoraStage = [string](Get-ConfigValue -Object $sam3Config -Name 'loraStage' -Default 'stage_a')
$sam3WarmupEnabled = [bool](Get-ConfigValue -Object $sam3Config -Name 'warmupEnabled' -Default $true)
$sam3MockDelayMs = [int](Get-ConfigValue -Object $sam3Config -Name 'mockDelayMs' -Default 0)
$sam3InferenceTimeoutSeconds = [int](Get-ConfigValue -Object $sam3Config -Name 'inferenceTimeoutSeconds' -Default 20)

$backendMode = if ($sam3Enabled) { 'sam3' } else { 'mock' }
$backendDir = Join-Path $rootPath 'Backend'
$frontendDir = Join-Path $rootPath 'Frontend'

if ($sam3Enabled -and $sam3RunImportCheck) {
    Write-Host '[SAM3 Check] Running check_sam3_import.py before startup' -ForegroundColor Yellow
    & $venvPython (Join-Path $rootPath 'check_sam3_import.py')
    if ($LASTEXITCODE -ne 0) {
        throw 'SAM3 import check failed. Startup aborted.'
    }
}

Write-Host 'Loaded startup configuration:' -ForegroundColor Green
Write-Host "  Backend  : enabled=$backendEnabled host=$backendHost port=$backendPort"
Write-Host "  DB Mode  : $backendDbBackend"
if (-not [string]::IsNullOrWhiteSpace($backendMysqlUrl)) {
    Write-Host '  Database : MYSQL_URL is provided by dev-launch.config.json'
} else {
    Write-Host '  Database : using Backend default MYSQL_URL (set backend.mysqlUrl to override)' -ForegroundColor Yellow
}
Write-Host "  Frontend : enabled=$frontendEnabled host=$frontendHost port=$frontendPort proxy=$proxyBackendTarget"
Write-Host "  SAM3     : enabled=$sam3Enabled mode=$backendMode device=$sam3Device"

if (-not $backendEnabled -and -not $frontendEnabled) {
    throw 'Both backend and frontend are disabled in config. Nothing to start.'
}

if ($backendEnabled) {
$mysqlEnvLine = ""
if (-not [string]::IsNullOrWhiteSpace($backendMysqlUrl)) {
    $mysqlEnvLine = "`$env:MYSQL_URL = '$backendMysqlUrl'"
}

$backendCommand = @"
`$env:DEBUG = 'false'
$mysqlEnvLine
`$env:DB_BACKEND = '$backendDbBackend'
`$env:MODEL_LOAD_MODE = '$backendMode'
`$env:MODEL_DEVICE = '$sam3Device'
`$env:MODEL_CHECKPOINT_PATH = '$sam3CheckpointPath'
`$env:MODEL_LORA_ENABLED = '$sam3LoraEnabled'
`$env:MODEL_LORA_PATH = '$sam3LoraPath'
`$env:MODEL_LORA_STAGE = '$sam3LoraStage'
`$env:MODEL_WARMUP_ENABLED = '$sam3WarmupEnabled'
`$env:MODEL_MOCK_DELAY_MS = '$sam3MockDelayMs'
`$env:MODEL_INFERENCE_TIMEOUT_SECONDS = '$sam3InferenceTimeoutSeconds'
Set-Location '$backendDir'
& '$venvPython' -m uvicorn app.main:app --host '$backendHost' --port $backendPort
"@

    Start-CommandWindow -Title 'Backend' -Command $backendCommand -IsDryRun:$DryRun
}

if ($frontendEnabled) {
    $frontendCommand = @"
`$env:VITE_BACKEND_PROXY_TARGET = '$proxyBackendTarget'
Set-Location '$frontendDir'
& '$npmCommand' run dev -- --host '$frontendHost' --port $frontendPort
"@

    Start-CommandWindow -Title 'Frontend' -Command $frontendCommand -IsDryRun:$DryRun
}

if ($DryRun) {
    Write-Host 'Dry run completed. No processes were started.' -ForegroundColor Yellow
} else {
    Write-Host 'Startup commands were launched in new PowerShell windows.' -ForegroundColor Green
}
