@echo off
cd /d C:\Users\tonyd\partswatchai
set PYEXE=python
where python >nul 2>&1 || set PYEXE=py
%PYEXE% scripts\adhoc\step1_discover.py > scripts\adhoc\step1_stdout.txt 2>&1
echo EXITCODE=%ERRORLEVEL% >> scripts\adhoc\step1_stdout.txt
echo DONE >> scripts\adhoc\step1_stdout.txt
