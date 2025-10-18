# CUDA + FreeGLUT + ImGui Demos

Interactive CUDA demos with real-time UI controls. Includes particle system, wave simulation, and Mandelbrot fractal explorer.

## Prerequisites

- **Windows 10/11** (64-bit)
- **Visual Studio 2019 or 2022** with C++ workload
- **CUDA Toolkit** (project uses 12.2, but 11.x or 12.x work)
- **NVIDIA GPU** with CUDA support

## Quick Setup

### 1. Install vcpkg and Libraries

```powershell
# Clone vcpkg
git clone https://github.com/Microsoft/vcpkg.git
cd vcpkg
.\bootstrap-vcpkg.bat

# Install dependencies (static libraries)
.\vcpkg install freeglut:x64-windows-static glew:x64-windows-static imgui[opengl3-binding,glut-binding]:x64-windows-static

# Integrate with Visual Studio
.\vcpkg integrate install
```

### 2. Update vcpkg Path (if needed)

If your vcpkg is not in `C:\Users\matth\GIT\vcpkg`, edit `CudaDemos.vcxproj`:
- Search for `C:\Users\matth\GIT\vcpkg\installed\x64-windows-static`
- Replace with your vcpkg path

### 3. Build and Run

1. Open `CudaDemos.sln` in Visual Studio
2. Set platform to **x64**
3. Build: **Ctrl+Shift+B**
4. Run: **Ctrl+F5**

## Controls

- **0-3**: Switch demos (0=Test, 1=Particles, 2=Waves, 3=Mandelbrot)
- **H**: Hide/show UI panel
- **ESC**: Exit
- **UI Panel**: Use buttons and sliders for interactive control

## Troubleshooting

### CUDA Version Mismatch
```
Error: CUDA 12.2.props not found
```
**Fix:** Edit `CudaDemos.vcxproj`, search for `CUDA 12.2` and replace with your version (e.g., `CUDA 11.8`)

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
2. Run `.\vcpkg integrate install`
3. Verify vcpkg path in `CudaDemos.vcxproj` matches your installation

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
demos.cu             - CUDA kernels
```

## License

Educational and demonstration purposes. Feel free to modify and extend.
