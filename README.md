# CUDA + OpenGL Demos

Interactive 3D graphics demos with real-time UI controls. Includes a rotating cube with Phong shading (OpenGL) and Mandelbrot fractal explorer (CUDA).

## Prerequisites

- **Windows 10/11** (64-bit)
- **Visual Studio 2019 or 2022** with C++ workload
- **CUDA Toolkit** (project uses 12.2, but 11.x or 12.x work)
- **NVIDIA GPU** with CUDA support

## Quick Setup

### 1. Install vcpkg and Libraries

**Important:** vcpkg must be installed at the same level as the `cuda-demos` folder.

```
parent-folder/
  cuda-demos/          <- This project
  vcpkg/              <- vcpkg installation (same level)
```

```powershell
# Navigate to the parent directory of cuda-demos
cd ..

# Clone vcpkg at the same level as cuda-demos
git clone https://github.com/Microsoft/vcpkg.git
cd vcpkg
.\bootstrap-vcpkg.bat

# Install dependencies (static libraries)
.\vcpkg install freeglut:x64-windows-static glew:x64-windows-static imgui[opengl3-binding,glut-binding]:x64-windows-static tinygltf:x64-windows-static

# Note: vcpkg integration is disabled for this project to avoid path conflicts.
# The project uses explicit relative paths to vcpkg libraries.
```

### 2. Build and Run

1. Open `CudaDemos.sln` in Visual Studio
2. Set platform to **x64**
3. Build: **Ctrl+Shift+B**
4. Run: **Ctrl+F5**

## Demos

### 1. 3D Boxes (OpenGL)
A rotating cube with Phong shading. Features:
- Real-time Phong lighting (ambient, diffuse, specular)
- Full 3D camera navigation (WASD + mouse)
- Adjustable rotation speed
- Adjustable light position and material properties

### 2. Mandelbrot Fractal (CUDA)
GPU-accelerated Mandelbrot set visualization. Features:
- Real-time zoom and pan with mouse
- Smooth color gradients
- Arrow key navigation

## Controls

### General
- **1**: Switch to 3D Boxes demo
- **2**: Switch to Mandelbrot Fractal demo
- **H**: Hide/show UI panel
- **ESC**: Exit

### 3D Camera (Boxes Demo only)
- **W/A/S/D**: Move forward/left/backward/right
- **Q/E**: Move down/up
- **Left Mouse Drag**: Orbit around clicked point
- **Middle Mouse Drag**: Pan/translate camera
- **Right Mouse Drag**: Rotate view (first-person)
- **Mouse Wheel**: Zoom in/out

### Mandelbrot Demo
- **Mouse Wheel**: Zoom
- **Click and Drag**: Pan
- **Arrow Keys**: Pan

## Troubleshooting

### CUDA Version Mismatch
```
Error: CUDA 12.2.props not found
```
**Fix:** Edit `CudaDemos.vcxproj`, search for `CUDA 12.2` and replace with your version (e.g., `CUDA 11.8`)

**Note:** The project uses `$(CUDA_PATH_V12_2)` instead of `$(CUDA_PATH)` to avoid conflicts when multiple CUDA versions are installed. If you're using a different version, replace all instances of `CUDA_PATH_V12_2` with your version (e.g., `CUDA_PATH_V11_8` or `CUDA_PATH_V12_8`)

### Build Tools Not Found
```
Error: The build tools for v142 cannot be found
```
**Fix:** Install VS 2019 build tools, or change `<PlatformToolset>v142</PlatformToolset>` to `v143` in the .vcxproj

### Libraries Not Found
```
Error: cannot open file 'imgui.lib' / 'glew32.lib' / 'freeglut.lib'
```
**Fix:** 
1. Ensure you installed static versions: `.\vcpkg install freeglut:x64-windows-static glew:x64-windows-static imgui[opengl3-binding,glut-binding]:x64-windows-static`
2. Verify vcpkg is installed at the same level as the cuda-demos folder (see setup instructions)
3. Check that the relative path `../vcpkg/installed/x64-windows-static/lib` is correct

### Runtime: GPU Not Found
```
Error: This project requires a CUDA capable GPU
```
**Fix:** Update NVIDIA drivers, verify GPU with `nvidia-smi` command

### Runtime: DLL Not Found
```
Error: glew32.dll or freeglut.dll not found
```
**Fix:** This shouldn't happen with static build. Verify `GLEW_STATIC` and `FREEGLUT_STATIC` are in preprocessor definitions.

## Project Files

```
CudaDemos.sln        - Visual Studio solution
CudaDemos.vcxproj    - Project configuration
main.cpp             - Application and UI code
BoxesDemo.h/cpp      - OpenGL 3D cube with Phong shading
MandelbrotDemo.h/cpp - Mandelbrot fractal
MandelbrotDemo.cu    - CUDA kernel for Mandelbrot
Demo.h               - Base demo interface
```

## License

Educational and demonstration purposes. Feel free to modify and extend.
