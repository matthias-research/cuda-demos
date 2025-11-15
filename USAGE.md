# Balls Demo - Quick Start Guide

## What is This?
A GPU-accelerated physics simulation showcasing **thousands of bouncing balls** with:
- Real-time collision detection
- Quaternion-based rotation
- Stripe patterns that spin with physics
- CUDA-OpenGL interoperability

## Controls

### Camera
- **W/A/S/D**: Move camera forward/left/back/right
- **Q/E**: Move camera down/up
- **Left Mouse**: Orbit around center
- **Middle Mouse**: Pan camera
- **Right Mouse**: Rotate view direction
- **Mouse Wheel**: Zoom in/out

### UI Parameters

#### Performance
- **FPS**: Real-time frame rate
- **Frame Time**: Milliseconds per frame
- **Ball Count**: Current number of balls

#### Physics
- **Ball Count** (100-10,000): Number of simulated balls
  - 1,000: Smooth on most GPUs
  - 5,000: Good mid-range GPU test
  - 10,000: High-end GPU showcase

- **Gravity** (0-20): Downward acceleration
  - Default: 9.8 (Earth gravity)
  - 0: Weightless (fun!)
  - 20: Heavy balls

- **Bounce** (0-1): Energy retention on collision
  - 0: Perfectly inelastic (no bounce)
  - 1: Perfectly elastic (bounces forever)
  - 0.85: Realistic default

- **Friction** (0.8-1.0): Velocity damping
  - 1.0: No friction
  - 0.99: Slight damping (default)
  - 0.95: High damping (slow motion)

- **Room Size** (5-30): Simulation box dimensions
  - 5: Cramped, lots of collisions
  - 10: Balanced default
  - 30: Spacious, fewer collisions

#### Lighting
- **Light X/Y/Z**: Position of main light source
  - Affects how stripe patterns are visible
  - Y controls height (higher = more even lighting)

### Buttons
- **Reset Simulation**: Restart with new random balls
- **Reset View**: Return camera to default position

## Visual Features

### Stripe Patterns
Each ball has alternating colored stripes that:
- Rotate based on angular velocity
- Show collision-induced spin
- Computed in fragment shader from quaternion

### Colors
Random vibrant colors per ball for easy tracking

### Sizes
Variable radii (4× larger than original demo) for visual interest

## Performance Tips

### Getting Best FPS
1. Start with 1,000 balls
2. If FPS > 200, increase count
3. If FPS < 30, decrease count
4. Optimal: Find highest count that maintains 60+ FPS

### Common FPS Ranges
| GPU Tier          | 1K balls | 5K balls | 10K balls |
|-------------------|----------|----------|-----------|
| Budget (GTX 1650) | 200+     | 40+      | 15+       |
| Mid (RTX 3060)    | 500+     | 100+     | 40+       |
| High (RTX 3080)   | 1000+    | 300+     | 100+      |
| Extreme (RTX 4090)| 2000+    | 800+     | 300+      |

### If Performance is Low
- Close other GPU-intensive apps
- Update NVIDIA drivers
- Lower ball count
- Check Windows TDR settings (if GPU keeps resetting)

## Fun Things to Try

### Zero Gravity Chaos
1. Set Gravity = 0
2. Set Ball Count = 5000
3. Watch the chaos!

### Pinball Machine
1. Set Room Size = 30
2. Set Gravity = 15
3. Set Bounce = 0.95
4. Balls bounce around like pinball!

### Slow Motion
1. Set Friction = 0.92
2. Watch balls gradually slow down
3. Like watching through syrup

### Dense Packing
1. Set Ball Count = 10000
2. Set Room Size = 8
3. Set Gravity = 20
4. Balls pack together at bottom

### Space Station
1. Set Gravity = 0
2. Set Room Size = 30
3. Give balls time to spread out
4. Looks like floating debris!

## Technical Notes

### Why is This Fast?
- **Spatial Hash Grid**: O(n) collision detection vs O(n²)
- **GPU Parallelism**: All balls updated simultaneously
- **Zero Copy**: Physics computed directly in render buffer
- **Cache Coherency**: Sorted by spatial proximity

### Hash Grid Visualization
If you could see the hash grid, it would be a 3D grid dividing space into cells. Each ball only checks its own cell + 26 neighbors (3×3×3 cube).

### When CPU Would Be Better
- Very few balls (< 20): GPU overhead not worth it
- Complex per-ball logic: GPU kernel launch overhead
- Need deterministic replay: GPU float math has variance

### Memory Usage
Approximate GPU memory per ball:
- VBO: 14 floats = 56 bytes
- Physics state: 64 bytes
- Hash data: 20 bytes
- **Total: ~140 bytes/ball**

10,000 balls ≈ 1.4 MB (negligible on modern GPUs)

## Troubleshooting

### Balls Not Moving
- Check if CUDA is enabled (green text in UI)
- Verify GPU is CUDA-capable
- Check console for CUDA errors

### Balls Pass Through Walls
- Room size might not match simulation bounds
- Try resetting simulation

### Stripes Not Rotating
- Quaternion integration might have issues
- Check if angular velocity is non-zero

### Flickering/Artifacts
- Z-fighting with floor plane
- Increase camera near plane distance

### Crashes on High Ball Count
- GPU memory exhausted
- Windows TDR timeout (GPU took too long)
- Solution: Lower ball count or increase TDR timeout

### Low FPS Despite Good GPU
- CPU-bound on framebuffer copy
- VSync enabled (limits to 60 FPS)
- Other GPU apps running

## Development Notes

See `GPU_IMPLEMENTATION.md` for technical details.
See `BUILD_CUDA.md` for compilation instructions.

## Credits

Physics: Classical mechanics + impulse-based collisions  
Rendering: OpenGL point sprites with billboarded spheres  
Rotation: Quaternion integration with angular velocity  
Optimization: CUDA spatial hash grid (Mitchell et al.)  
Libraries: CUDA, Thrust, OpenGL, GLEW, ImGui

Enjoy the bouncing balls! 🎱

