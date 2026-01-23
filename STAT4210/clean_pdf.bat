@echo off
REM Usage: export_pdf.bat demo.ipynb

if "%~1"=="" (
    echo Usage: clean_pdf.bat notebook.ipynb
    exit /b 1
)

set NOTEBOOK=%~1

jupyter nbconvert "%NOTEBOOK%" ^
    --to pdf ^
    --TemplateExporter.exclude_markdown=True

if errorlevel 1 (
    echo Export failed.
    exit /b 1
)

echo Export succeeded.