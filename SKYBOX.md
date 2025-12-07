# Skybox System

## Overview
The skybox system provides a simple way to add background environment rendering to 3D demos. The skybox is always rendered behind all other geometry.

## Files
- `Skybox.h` - Header with class definition
- `Skybox.cpp` - Implementation with BMP loader and rendering

## Usage

### 1. Include the header
```cpp
#include "Skybox.h"
```

### 2. Add member variable to your demo
```cpp
class MyDemo : public Demo {
private:
    Skybox* skybox = nullptr;
    bool showSkybox = true;  // Optional toggle
    // ...
};
```

### 3. Initialize in constructor
```cpp
MyDemo::MyDemo() {
    // ... other initialization ...
    
    skybox = new Skybox();
    if (!skybox->loadFromBMP("assets/skybox.bmp")) {
        delete skybox;
        skybox = nullptr;
    }
}
```

### 4. Render in render3D()
```cpp
void MyDemo::render3D(int width, int height) {
    // Clear buffers
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    
    // Render skybox first (always in background)
    if (skybox) {
        skybox->render(camera);
    }
    
    // ... render other 3D content ...
}
```

### 5. Cleanup in destructor
```cpp
MyDemo::~MyDemo() {
    if (skybox) {
        skybox->cleanup();
        delete skybox;
    }
    // ... other cleanup ...
}
```

## BMP Format Requirements

### Cubemap Cross Layout
The BMP file must be in a **cubemap cross** layout with the following arrangement:

```
      [top]
[left][front][right][back]
     [bottom]
```

### Format Specifications
- **File format**: BMP (uncompressed)
- **Bit depth**: 24-bit RGB
- **Dimensions**: Must be 4:3 ratio (width = 4 × faceSize, height = 3 × faceSize)
- **Face size**: Automatically detected from image dimensions
- **Recommended sizes**: 512×384, 1024×768, 2048×1536

### Example Dimensions
- Face size 256: Image is 1024 × 768
- Face size 512: Image is 2048 × 1536
- Face size 1024: Image is 4096 × 3072

## How It Works

### Rendering
1. Skybox cube is rendered at the camera position (no translation)
2. Only camera rotation is applied
3. Depth is set to maximum (1.0) in the vertex shader
4. This ensures the skybox is always behind all other geometry

### Shaders
- **Vertex shader**: Removes translation from view matrix, forces depth to 1.0
- **Fragment shader**: Samples cubemap using direction vector

### Face Mapping
The loader automatically extracts the 6 cube faces from the cross layout:
- **+X (right)**: Position [1, 1] in grid
- **-X (left)**: Position [3, 1] in grid (back in cross)
- **+Y (top)**: Position [1, 0] in grid
- **-Y (bottom)**: Position [1, 2] in grid
- **+Z (front)**: Position [2, 1] in grid (right in cross)
- **-Z (back)**: Position [0, 1] in grid (left in cross)

## Creating Skybox Textures

### From Existing Images
1. Find or create 6 square images (one for each cube face)
2. Arrange them in the cross layout
3. Save as 24-bit BMP

### Tools
- **Photoshop/GIMP**: Manually arrange faces
- **Skybox generators**: Many online tools create cubemap crosses
- **HDR to cubemap converters**: Convert HDRI to cubemap format

### Online Resources
- Skybox images: [polyhaven.com](https://polyhaven.com/hdris)
- Cubemap tools: Search for "cubemap cross generator"

## Example: BallsDemo Integration

See `BallsDemo.h` and `BallsDemo.cpp` for a complete integration example:
- Skybox loaded in constructor
- Rendered at the start of `render3D()`
- Toggle checkbox in ImGui UI
- Proper cleanup in destructor
