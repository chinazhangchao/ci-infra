@echo off
call "%VLLM_WINDOWS_VCVARSALL%" x64
if errorlevel 1 exit /b %errorlevel%
if /i not "%VSCMD_ARG_TGT_ARCH%"=="x64" (
    echo ERROR: Visual Studio did not select the x64 compiler.
    exit /b 1
)
"%VLLM_WINDOWS_BUILD_PYTHON%" -m build --wheel --no-isolation --skip-dependency-check --outdir "%VLLM_WINDOWS_WHEEL_DIR%"
exit /b %errorlevel%
