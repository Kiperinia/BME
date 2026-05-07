# BME
## 使用了triton-3.0.0-cp312-cp312-win_amd64.whl以在Windows上运行
运行前先使用check_sam3_import.py检查是否可以使用SAM3，linux系统自行替换triton

## 根目录一键启动

根目录新增了以下文件：

- dev-launch.config.json：统一控制前端、后端和 SAM3 启动参数
- start-dev.ps1：PowerShell 一键启动脚本
- start-dev.cmd：Windows 双击入口

默认启动方式：

```powershell
Set-Location E:\BME
./start-dev.ps1
```

如果想先检查配置而不真正启动进程：

```powershell
Set-Location E:\BME
./start-dev.ps1 -DryRun
```

SAM3 可选开关位于 dev-launch.config.json：

```json
"sam3": {
	"enabled": false,
	"runImportCheck": false,
	"device": "cuda",
	"checkpointPath": "MedicalSAM3/checkpoint/MedSAM3.pt"
}
```

说明：

- sam3.enabled=false 时，后端以 mock 模式启动
- sam3.enabled=true 时，后端以真实 SAM3 模式启动
- sam3.runImportCheck=true 时，启动前会先运行 check_sam3_import.py，检查失败则终止启动
- 前端会自动读取 frontend.proxyBackendTarget 作为 /api 代理目标