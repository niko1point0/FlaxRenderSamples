@echo off
setlocal

for /d %%F in (*) do (
    if exist "%%F\Binaries" (
        echo Deleting %%F\Binaries
        rmdir /s /q "%%F\Binaries"
    )
    if exist "%%F\Cache" (
        echo Deleting %%F\Cache
        rmdir /s /q "%%F\Cache"
    )
    if exist "%%F\Logs" (
        echo Deleting %%F\Logs
        rmdir /s /q "%%F\Logs"
    )
    if exist "%%F\Output" (
        echo Deleting %%F\Output
        rmdir /s /q "%%F\Output"
    )
)

echo Done.
pause