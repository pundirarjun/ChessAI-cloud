@echo off
rem Local Windows dev environment for building az_cpp_mcts: vcvars64 + the
rem NuGet-extracted Windows SDK (headers/libs/tools) laid out under %WINSDK_NUGET%.
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
if errorlevel 1 exit /b 1
set "WINSDK_NUGET=C:\Users\HP\AppData\Local\Temp\opencode\winsdk"
set "SDKINC=%WINSDK_NUGET%\sdkcpp\c\Include\10.0.26100.0"
set "SDKLIB=%WINSDK_NUGET%\sdkcppx64\c"
set "SDKBIN=%WINSDK_NUGET%\buildtools\bin\10.0.26100.0\x64"
set "INCLUDE=%SDKINC%\ucrt;%SDKINC%\um;%SDKINC%\shared;%SDKINC%\winrt;%SDKINC%\cppwinrt;%INCLUDE%"
set "LIB=%SDKLIB%\um\x64;%SDKLIB%\ucrt\x64;%LIB%"
set "PATH=%SDKBIN%;%PATH%"
