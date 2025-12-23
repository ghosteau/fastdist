param(
    [string[]]$PythonVersion = @("3.12", "3.13", "3.14"),
    [switch]$Clean
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

    # Build sdist and wheel
    python setup.py sdist bdist_wheel --dist-dir $distRoot

    deactivate

    # Cleaning up the file directory
    if ($Clean)
    {
        Remove-Item -Recurse -Force -LiteralPath $venvPath
        $tempDirs = @("build", "fastdist.egg-info") | ForEach-Object { Join-Path $projectRoot $_ }
        foreach ($dir in $tempDirs)
        {
            if (Test-Path $dir)
            {
                Remove-Item -Recurse -Force -LiteralPath $dir
            }
        }
    }
}

Write-Host "All builds complete. Wheels are in $distDir"
