# CUDA + FreeGLUT Demos Setup Script
# Run this script in PowerShell to check your environment

Write-Host "CUDA + FreeGLUT Demos - Environment Check" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# Check for Visual Studio
Write-Host "Checking for Visual Studio..." -ForegroundColor Yellow
$vsPath = Get-ChildItem "C:\Program Files*\Microsoft Visual Studio" -ErrorAction SilentlyContinue -Directory | 
    Get-ChildItem -Filter "*" -ErrorAction SilentlyContinue -Directory |
    Where-Object { Test-Path (Join-Path $_.FullName "Common7\IDE\devenv.exe") } |
    Select-Object -First 1

if ($vsPath) {
    Write-Host "✓ Visual Studio found: $($vsPath.FullName)" -ForegroundColor Green
} else {
    Write-Host "✗ Visual Studio not found!" -ForegroundColor Red
    Write-Host "  Please install Visual Studio 2019 or newer with C++ development tools" -ForegroundColor Red
}

# Check for CUDA
Write-Host ""
Write-Host "Checking for CUDA Toolkit..." -ForegroundColor Yellow
$cudaPath = $env:CUDA_PATH
if ($cudaPath -and (Test-Path $cudaPath)) {
    Write-Host "✓ CUDA Toolkit found: $cudaPath" -ForegroundColor Green
    $nvccPath = Join-Path $cudaPath "bin\nvcc.exe"
    if (Test-Path $nvccPath) {
        $version = & $nvccPath --version 2>&1 | Select-String "release" | ForEach-Object { $_.ToString() }
        Write-Host "  Version: $version" -ForegroundColor Green
    }
} else {
    Write-Host "✗ CUDA Toolkit not found!" -ForegroundColor Red
    Write-Host "  Please install CUDA Toolkit from: https://developer.nvidia.com/cuda-downloads" -ForegroundColor Red
    Write-Host "  You may need to restart PowerShell after installation" -ForegroundColor Yellow
}

# Check for NVIDIA GPU
Write-Host ""
Write-Host "Checking for NVIDIA GPU..." -ForegroundColor Yellow
try {
    $gpu = Get-WmiObject Win32_VideoController | Where-Object { $_.Name -like "*NVIDIA*" }
    if ($gpu) {
        Write-Host "✓ NVIDIA GPU found: $($gpu.Name)" -ForegroundColor Green
    } else {
        Write-Host "✗ No NVIDIA GPU detected!" -ForegroundColor Red
        Write-Host "  CUDA demos require an NVIDIA GPU with CUDA support" -ForegroundColor Red
    }
} catch {
    Write-Host "✗ Unable to detect GPU information" -ForegroundColor Red
}

# Check for vcpkg
Write-Host ""
Write-Host "Checking for vcpkg (optional)..." -ForegroundColor Yellow
$vcpkgPath = $env:VCPKG_ROOT
if ($vcpkgPath -and (Test-Path (Join-Path $vcpkgPath "vcpkg.exe"))) {
    Write-Host "✓ vcpkg found: $vcpkgPath" -ForegroundColor Green
    
    # Check if freeglut is installed
    $vcpkgExe = Join-Path $vcpkgPath "vcpkg.exe"
    $installed = & $vcpkgExe list freeglut:x64-windows 2>&1
    if ($installed -match "freeglut") {
        Write-Host "✓ FreeGLUT is installed via vcpkg" -ForegroundColor Green
    } else {
        Write-Host "✗ FreeGLUT not installed via vcpkg" -ForegroundColor Yellow
        Write-Host "  To install: vcpkg install freeglut:x64-windows" -ForegroundColor Yellow
    }
} else {
    Write-Host "○ vcpkg not found (optional)" -ForegroundColor Gray
    Write-Host "  You can install FreeGLUT manually or use vcpkg" -ForegroundColor Gray
}

# Summary
Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Setup Summary" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next Steps:" -ForegroundColor Yellow
Write-Host "1. Ensure all required components are installed (marked with ✓)" -ForegroundColor White
Write-Host "2. Install FreeGLUT using vcpkg or manually" -ForegroundColor White
Write-Host "3. Open CudaDemos.sln in Visual Studio" -ForegroundColor White
Write-Host "4. Build and run the project (Ctrl+F5)" -ForegroundColor White
Write-Host ""
Write-Host "For more information, see README.md" -ForegroundColor Gray

