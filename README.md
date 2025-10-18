# CUDA + FreeGLUT Demos

A minimal collection of interactive CUDA demos using FreeGLUT and OpenGL for visualization on Windows.

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
- **FreeGLUT** and **GLEW** (statically linked)
  - Download from: https://freeglut.sourceforge.net/ and https://glew.sourceforge.net/
  - Or use vcpkg: `vcpkg install freeglut:x64-windows-static glew:x64-windows-static`

### Installing FreeGLUT and GLEW (Static Libraries)

#### Option 1: Using vcpkg (Recommended)

```powershell
# Install vcpkg if you haven't already
git clone https://github.com/Microsoft/vcpkg.git
cd vcpkg
.\bootstrap-vcpkg.bat

# Install required libraries (static versions - no DLLs needed!)
.\vcpkg install freeglut:x64-windows-static glew:x64-windows-static

# Integrate with Visual Studio
.\vcpkg integrate install
```

**Note:** The project uses **static libraries** so you don't need any DLLs. Everything is compiled into the executable.

#### Option 2: Manual Installation (Advanced)

1. Download FreeGLUT and GLEW **static library** versions for Windows
2. Build as static libraries or download pre-built static versions
3. Update `CudaDemos.vcxproj`:
   - Add include directories to `AdditionalIncludeDirectories`
   - Add library directories to `AdditionalLibraryDirectories`
   - Make sure `GLEW_STATIC` and `FREEGLUT_STATIC` are defined in preprocessor definitions

**Note:** Using vcpkg is much easier and handles all paths automatically!

## Building the Project

### Using Visual Studio

1. Open `CudaDemos.sln` in Visual Studio
2. Make sure the platform is set to **x64** (top toolbar)
3. Select **Debug** or **Release** configuration
4. Build the solution: **Build → Build Solution** (or press `Ctrl+Shift+B`)
5. Run: **Debug → Start Without Debugging** (or press `Ctrl+F5`)

### Important Notes

- **The project uses static libraries (no DLLs required)** - FreeGLUT and GLEW are compiled into the executable
  - Preprocessor defines: `GLEW_STATIC` and `FREEGLUT_STATIC`
  - Uses vcpkg's `x64-windows-static` triplet

- **The project is configured for CUDA 12.2 and Visual Studio 2019 (v142) toolset** - compatible with VS 2019 and VS 2022
  
- If you have a different CUDA version, edit `CudaDemos.vcxproj`:
  - Find and replace `CUDA 12.2` with your version (e.g., `CUDA 11.8`, `CUDA 12.3`, etc.)
  - Look for lines containing `BuildCustomizations\CUDA 12.2`
  
- The vcpkg paths are hardcoded to `C:\Users\matth\GIT\vcpkg\installed\x64-windows-static`
  - If your vcpkg is in a different location, update the paths in `CudaDemos.vcxproj`
  - Search for `vcpkg\installed` and replace with your vcpkg path

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
├── CudaDemos.vcxproj          # Project file with CUDA configuration
├── CudaDemos.vcxproj.filters  # Project filters for organization
├── main.cpp                   # Main application with OpenGL/GLEW/CUDA interop
├── demos.cu                   # CUDA kernels (Particles, Waves, Mandelbrot)
├── setup.ps1                  # Environment check script
├── .gitignore                 # Git ignore rules for VS and CUDA
└── README.md                  # This file
```

## Troubleshooting

### Build Errors

**"Cannot find CUDA Toolkit"** or **"CUDA 12.2.props not found"**
- Ensure CUDA is installed and the `CUDA_PATH` environment variable is set
- If you have a different CUDA version, edit `CudaDemos.vcxproj` and replace `12.2` with your version
- Restart Visual Studio after CUDA installation

**"The build tools for v142 cannot be found"**
- Install the Visual Studio 2019 build tools component
- Or edit `CudaDemos.vcxproj` and change `<PlatformToolset>v142</PlatformToolset>` to `v143` (VS 2022)

**"freeglut.lib or glew32.lib not found"**
- Make sure you installed the **static** versions: `vcpkg install freeglut:x64-windows-static glew:x64-windows-static`
- Then ran: `vcpkg integrate install`
- Check that the vcpkg path in `CudaDemos.vcxproj` matches your vcpkg installation location
- The project looks for libraries in: `C:\Users\matth\GIT\vcpkg\installed\x64-windows-static\lib`

**"glew32.dll or freeglut.dll not found"**
- This shouldn't happen - the project uses static libraries (no DLLs needed)
- If you see this error, make sure `GLEW_STATIC` and `FREEGLUT_STATIC` are defined in preprocessor definitions

### Runtime Errors

**"This project requires a CUDA capable GPU"**
- Verify you have an NVIDIA GPU with CUDA support
- Update your NVIDIA drivers to the latest version
- Check Device Manager to ensure GPU is recognized

**Black screen or application crashes**
- Update your graphics drivers to the latest version
- Try the Debug build for more detailed error information
- Check Visual Studio Output window for CUDA errors
- Verify your GPU has CUDA support by running `nvidia-smi` in command prompt

## Customizing Demos

All CUDA kernels are in `demos.cu` - only **~180 lines** of GPU code! Each demo is a simple kernel function:
- `particleKernel` - Modify particle behavior and colors
- `waveKernel` - Adjust wave parameters and interference patterns
- `mandelbrotKernel` - Change fractal parameters and coloring

The main application in `main.cpp` handles:
- OpenGL window setup with FreeGLUT
- GLEW initialization for OpenGL extensions
- CUDA-OpenGL interoperability (pixel buffer objects)
- Keyboard controls and demo switching

The code is designed to be minimal and easy to understand. Feel free to experiment and add your own demos!

## License

This project is provided as-is for educational and demonstration purposes.

## Quick Start Summary

1. **Install Prerequisites:**
   - Visual Studio 2019 or 2022 with C++ support
   - CUDA Toolkit (project configured for 12.2, but any 11.x or 12.x works)
   - NVIDIA GPU with CUDA support

2. **Install vcpkg and Libraries (Static - No DLLs!):**
   ```powershell
   git clone https://github.com/Microsoft/vcpkg.git
   cd vcpkg
   .\bootstrap-vcpkg.bat
   .\vcpkg install freeglut:x64-windows-static glew:x64-windows-static
   .\vcpkg integrate install
   ```

3. **Update vcpkg Path (if needed):**
   - If your vcpkg is not in `C:\Users\matth\GIT\vcpkg`, edit `CudaDemos.vcxproj`
   - Search for `vcpkg\installed\x64-windows-static` and replace with your path

4. **Build and Run:**
   - Open `CudaDemos.sln` in Visual Studio
   - Set platform to **x64**
   - Build (Ctrl+Shift+B) and Run (Ctrl+F5)

5. **Enjoy the demos!** Press 1, 2, 3 to switch between them.

## Why Static Libraries?

The project uses **static linking** for FreeGLUT and GLEW, which means:
- ✅ **No DLL dependencies** - Everything is compiled into the executable
- ✅ **Easier distribution** - Just share the .exe file (plus CUDA runtime on target system)
- ✅ **No DLL version conflicts** - The exact library versions are baked in
- ⚠️ **Larger executable** - About 2-3 MB larger than dynamic version
- ⚠️ **Longer compile time** - But only on first build

