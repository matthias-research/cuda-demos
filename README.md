# CUDA + FreeGLUT Demos

A minimal collection of interactive CUDA demos using FreeGLUT for visualization on Windows.

## Demos Included

1. **Particle System** - Animated particles with dynamic colors
2. **Wave Simulation** - Rippling wave interference patterns
3. **Mandelbrot Fractal** - Interactive fractal explorer with zoom and pan

## Prerequisites

### Required Software

- **Windows 10/11** (64-bit)
- **Visual Studio 2022** (or 2019)
  - Desktop development with C++ workload
- **NVIDIA GPU** with CUDA support
- **CUDA Toolkit 12.x** (or 11.x)
  - Download from: https://developer.nvidia.com/cuda-downloads
  - Make sure to install the Visual Studio integration
- **FreeGLUT**
  - Download from: https://freeglut.sourceforge.net/
  - Or use vcpkg: `vcpkg install freeglut:x64-windows`

### Installing FreeGLUT

#### Option 1: Using vcpkg (Recommended)

```powershell
# Install vcpkg if you haven't already
git clone https://github.com/Microsoft/vcpkg.git
cd vcpkg
.\bootstrap-vcpkg.bat

# Install freeglut
.\vcpkg install freeglut:x64-windows

# Integrate with Visual Studio
.\vcpkg integrate install
```

#### Option 2: Manual Installation

1. Download FreeGLUT binaries for Windows
2. Extract and copy:
   - `freeglut.dll` to your Windows\System32 folder (or project directory)
   - `freeglut.lib` to a library folder
   - Header files to an include folder
3. Update the project properties to include these paths

## Building the Project

### Using Visual Studio

1. Open `CudaDemos.sln` in Visual Studio
2. Make sure the platform is set to **x64** (top toolbar)
3. Select **Debug** or **Release** configuration
4. Build the solution: **Build → Build Solution** (or press `Ctrl+Shift+B`)
5. Run: **Debug → Start Without Debugging** (or press `Ctrl+F5`)

### Important Build Notes

- If you get CUDA version errors, edit `CudaDemos.vcxproj`:
  - Replace `CUDA 12.6` with your installed version (e.g., `CUDA 12.3`, `CUDA 11.8`)
  - Look for lines containing `BuildCustomizations\CUDA 12.6`
  
- If you installed FreeGLUT manually (not via vcpkg):
  - Right-click project → Properties
  - Add include directories under: **C/C++ → General → Additional Include Directories**
  - Add library directories under: **Linker → General → Additional Library Directories**

## Controls

### General
- **1** - Switch to Particle System demo
- **2** - Switch to Wave Simulation demo
- **3** - Switch to Mandelbrot Fractal demo
- **ESC** - Exit application

### Mandelbrot Demo
- **Arrow Keys** - Pan around the fractal
- **+/-** - Zoom in/out

## Project Structure

```
cuda-demos/
├── CudaDemos.sln              # Visual Studio solution
├── CudaDemos.vcxproj          # Project file
├── CudaDemos.vcxproj.filters  # Project filters
├── main.cpp                   # Main application and OpenGL setup
├── demos.cu                   # CUDA kernels for all demos
└── README.md                  # This file
```

## Troubleshooting

### "Cannot find CUDA Toolkit"
- Ensure CUDA is installed and the `CUDA_PATH` environment variable is set
- Restart Visual Studio after CUDA installation

### "freeglut.lib not found"
- If using vcpkg, make sure you ran `vcpkg integrate install`
- If manual installation, check library paths in project properties

### "This project requires a CUDA capable GPU"
- Verify you have an NVIDIA GPU with CUDA support
- Update your NVIDIA drivers to the latest version

### Black screen or application crashes
- Update your graphics drivers
- Try the Debug build for more error information
- Check Visual Studio Output window for CUDA errors

## Customizing Demos

All CUDA kernels are in `demos.cu`. Each demo is a simple kernel function:
- `particleKernel` - Modify particle behavior
- `waveKernel` - Adjust wave parameters
- `mandelbrotKernel` - Change fractal parameters

The code is designed to be minimal and easy to understand. Feel free to experiment!

## License

This project is provided as-is for educational and demonstration purposes.

## Requirements Summary

- ✅ Windows 10/11 (64-bit)
- ✅ Visual Studio 2019 or newer
- ✅ CUDA Toolkit 11.x or 12.x
- ✅ FreeGLUT library
- ✅ NVIDIA GPU with CUDA support

