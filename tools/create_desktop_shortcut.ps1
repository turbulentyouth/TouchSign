$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root '.venv-runtime\Scripts\pythonw.exe'
if (-not (Test-Path -LiteralPath $python)) { throw 'Install requirements-desktop.txt in .venv-runtime first.' }
New-Item -ItemType Directory -Force -Path (Join-Path $root 'dist') | Out-Null
$shell = New-Object -ComObject WScript.Shell
$shortcut = $shell.CreateShortcut((Join-Path $root 'dist\TouchSign-Desktop.lnk'))
$shortcut.TargetPath = $python
$shortcut.Arguments = '"' + (Join-Path $root 'desktop_main.py') + '"'
$shortcut.WorkingDirectory = $root
$shortcut.Description = 'TouchSign independent embedded desktop window'
$shortcut.Save()
Write-Output 'Ready: dist\TouchSign-Desktop.lnk (requires this project and its runtime environment)'
