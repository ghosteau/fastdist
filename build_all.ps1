param(
    [string[]]$PythonVersion = @("3.12", "3.13", "3.14"),
    [switch]$Clean,
    [switch]$EnableCuda,
    [string]$PipInstall
)

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$distRoot = Join-Path $projectRoot "dist"

# Create dist folder
$distDir = Join-Path $projectRoot "dist"
if (-Not (Test-Path $distDir))
{
    New-Item -ItemType Directory -Path $distDir | Out-Null
}

foreach ($version in $PythonVersion)
{
    Write-Host "=== Building for Python $version ==="

    # Finding python executable
    $pythonExe = & py -$version -c "import sys; print(sys.executable)" 2> $null
    if (-not $pythonExe)
    {
        Write-Warning "Python $version not found. Skipping."
        continue
    }

    $venvPath = Join-Path $projectRoot ".venv$version"

    # Create venv if it doesn't exist
    if (Test-Path $venvPath)
    {
        Remove-Item -Recurse -Force $venvPath
    }
    & $pythonExe -m venv $venvPath

    $activateScript = Join-Path $venvPath "Scripts\Activate.ps1"

    # Activate venv
    & $activateScript

    # Upgrade pip and install build dependencies
    python.exe -m pip install --upgrade pip
    pip install setuptools wheel pybind11 -r (Join-Path $projectRoot "requirements.txt") -q

    # Set environment variable for CUDA
    if ($EnableCuda)
    {
        $env:FASTDIST_ENABLE_CUDA = "1"
        Write-Host "CUDA support: ENABLED" -ForegroundColor Green
    }
    else
    {
        $env:FASTDIST_ENABLE_CUDA = "0"
        Write-Host "CUDA support: DISABLED" -ForegroundColor Yellow
    }

    # Build wheel
    python setup.py sdist bdist_wheel --dist-dir $distRoot

    if ($LASTEXITCODE -ne 0)
    {
        Write-Error "Build failed for Python $version (exit code $LASTEXITCODE)"
        deactivate
        exit
    }

    deactivate

    # Cleaning up the file directory
    if ($Clean)
    {
        Write-Host "Deleting Build Files..." -ForegroundColor Yellow
        Remove-Item -Recurse -Force -LiteralPath $venvPath
        $tempDirs = @("build", "python/fastdist.egg-info") | ForEach-Object { Join-Path $projectRoot $_ }
        foreach ($dir in $tempDirs)
        {
            if (Test-Path $dir)
            {
                Remove-Item -Recurse -Force -LiteralPath $dir
            }
        }
    }

    Write-Host "Build for Python $version complete." -ForegroundColor Green
}

Write-Host "All builds complete. Wheels are in $distDir"
Write-Host "----------------------------------------------------"

if ($PipInstall)
{
    Write-Host "Installing built wheel for Python $PipInstall via pip..." -ForegroundColor Yellow
    $installVersion = $PipInstall

    if ($PythonVersion -notcontains $installVersion)
    {
        Write-Warning "The version specified for installation (`$installVersion`) was not included in the build loop). Aborting installation."
    }
    else
    {
        $versionTag = $installVersion -replace '\.', ''
        $searchFilter = "*cp$( $versionTag -replace '\.', '' )*.whl"
        $wheelFile = Get-ChildItem -Path $distRoot -Filter $searchFilter | Sort-Object LastWriteTime -Descending | Select-Object -First 1
        if ($wheelFile)
        {
            Write-Host "Found wheel: $( $wheelFile.Name )" -ForegroundColor Green

            $venvPath = Join-Path $projectRoot ".venv"
            $activateScript = Join-Path $venvPath "Scripts\Activate.ps1"

            & $activateScript

            Write-Host "Installing $( $wheelFile.Name ) into .venv..." -ForegroundColor Yellow
            pip install $wheelFile.FullName --force-reinstall

            deactivate
            Write-Host "Installation complete." -ForegroundColor Cyan
        }
        else
        {
            Write-Warning "No wheel file found for Python $installVersion in $distDir"
        }
    }

    if ($Clean -and (Test-Path "/python/fastdist.egg-info"))
    {
        $dir = Join-Path $projectRoot "python/fastdist.egg-info"
        Write-Host "Deleting fastdist.egg-info from fastdits installation..." -ForegroundColor Yellow
        Remove-Item -Recurse -Force -LiteralPath $dir
    }
}