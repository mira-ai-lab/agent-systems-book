# 书稿仓库开发环境一键安装（PowerShell）
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

Write-Host "Installing agent-platform (framework + services + supervisor + a2a)..."
pip install -e ".[api,dev,supervisor,a2a]"

Write-Host "Installing agent-platform-domains-builtin (entry_points)..."
pip install -e domains/

Write-Host "Done. Run: pytest"
