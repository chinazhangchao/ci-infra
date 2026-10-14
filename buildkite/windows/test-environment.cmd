@echo off
call "%VLLM_TEST_VCVARS%" %VLLM_TEST_VCVARS_ARGS%
if errorlevel 1 exit /b %errorlevel%
echo __VLLM_TEST_ENVIRONMENT__
set
