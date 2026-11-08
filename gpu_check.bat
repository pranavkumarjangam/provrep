@echo off
cd /d "%~dp0"
echo Checking GPU... (writes gpu_info.txt next to this file)
> gpu_info.txt echo ==== Video controllers ====
powershell -NoProfile -Command "Get-CimInstance Win32_VideoController | Select-Object Name, DriverVersion, @{n='VRAM_GB';e={[math]::Round($_.AdapterRAM/1GB,1)}} | Format-List" >> gpu_info.txt 2>&1
>> gpu_info.txt echo ==== nvidia-smi ====
nvidia-smi >> gpu_info.txt 2>&1
>> gpu_info.txt echo ==== CPU / RAM ====
powershell -NoProfile -Command "Get-CimInstance Win32_Processor | Select-Object Name, NumberOfCores, NumberOfLogicalProcessors | Format-List; 'RAM_GB: ' + [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory/1GB,1)" >> gpu_info.txt 2>&1
type gpu_info.txt
echo.
echo Done. Tell Claude "gpu checked".
pause
