#Requires -RunAsAdministrator

Set-ItemProperty 'HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem' `
    -Name 'LongPathsEnabled' `
    -Value 1 `
    -Type DWord

Write-Host "Long paths are now enabled on this machine!"
