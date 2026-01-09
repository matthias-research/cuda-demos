"""
Blender script to import ball positions and create instances using Geometry Nodes.
Select the mesh object you want to use as instance template, then run this script.
This uses Geometry Nodes for efficient handling of millions of instances.
"""

import bpy
import struct
import os
import random
from mathutils import Euler

def import_balls_geonodes(filepath):
    """Import ball data from binary file and create instances using Geometry Nodes."""
    
    # Get selected object (should be the mesh to instance)
    selected_objects = bpy.context.selected_objects
    if not selected_objects:
        print("Error: Please select a mesh object to use as instance template")
        return
    
    template_obj = selected_objects[0]
    if template_obj.type != 'MESH':
        print("Error: Selected object must be a mesh")
        return
    
    print(f"Using '{template_obj.name}' as instance template")
    
    # Check if file exists
    if not os.path.exists(filepath):
        print(f"Error: File not found: {filepath}")
        return
    
    # Read binary file
    with open(filepath, 'rb') as f:
        # Read number of balls
        num_balls_data = f.read(4)
        if len(num_balls_data) < 4:
            print("Error: Invalid file format")
            return
        
        num_balls = struct.unpack('I', num_balls_data)[0]
        print(f"Loading {num_balls} balls...")
        
        # Create a mesh for the point cloud
        mesh = bpy.data.meshes.new("Balls_PointCloud")
        
        # Allocate vertices and read data
        verts = []
        radii = []
        
        for i in range(num_balls):
            # Read position (3 floats) and radius (1 float)
            data = f.read(16)  # 4 floats * 4 bytes
            if len(data) < 16:
                print(f"Warning: Unexpected EOF at ball {i}")
                break
            
            x, y, z, radius = struct.unpack('ffff', data)
            verts.append((x, y, z))
            radii.append(radius)
            
            # Print progress every 100000 balls
            if (i + 1) % 100000 == 0:
                print(f"  Read {i + 1} / {num_balls} balls...")
        
        # Create mesh from verts
        print("Creating mesh from vertices...")
        mesh.from_pydata(verts, [], [])
        mesh.update()
        
        # Add radius as vertex attribute (for reference, though not used by instances)
        print("Adding radius attribute...")
        attr = mesh.attributes.new(name="radius", type='FLOAT', domain='POINT')
        for i, radius_val in enumerate(radii):
            attr.data[i].value = radius_val
        
        # Create collection for instances
        instances_collection = bpy.data.collections.new("BallInstances")
        bpy.context.scene.collection.children.link(instances_collection)
        
        # Create instances directly
        print("Creating instances...")
        for i in range(len(verts)):
            # Create instance
            inst_obj = bpy.data.objects.new(f"Ball_{i}", template_obj.data)
            instances_collection.objects.link(inst_obj)
            
            # Convert from OpenGL (Y-up) to Blender (Z-up) coordinate system
            # OpenGL: X=right, Y=up, Z=toward viewer
            # Blender: X=right, Y=back, Z=up
            # Transformation: X_bl=X_gl, Y_bl=-Z_gl, Z_bl=Y_gl
            x_gl, y_gl, z_gl = verts[i]
            x_bl = x_gl
            y_bl = -z_gl
            z_bl = y_gl
            
            # Set position and scale
            inst_obj.location = (x_bl, y_bl, z_bl)
            inst_obj.scale = (radii[i], radii[i], radii[i])
            
            # Assign random rotation
            random_euler = Euler((
                random.uniform(0, 2 * 3.14159),  # X rotation
                random.uniform(0, 2 * 3.14159),  # Y rotation
                random.uniform(0, 2 * 3.14159)   # Z rotation
            ), 'XYZ')
            inst_obj.rotation_euler = random_euler
            
            # Print progress every 100000 balls
            if (i + 1) % 100000 == 0:
                print(f"  Created {i + 1} / {len(verts)} instances...")
        
        print(f"\nSuccessfully created {len(verts)} ball instances!")
        print(f"Instances collection: {instances_collection.name}")
        print("Each instance is scaled by its radius value")

# Main execution
if __name__ == "__main__":
    # Try to find balls.bin in multiple locations
    script_dir = os.path.dirname(os.path.abspath(__file__))
    possible_paths = [
        "balls.bin",
        os.path.join(script_dir, "balls.bin"),
        "d:/balls.bin",
        "d:\\balls.bin",
    ]
    
    filepath = None
    for path in possible_paths:
        if os.path.exists(path):
            filepath = path
            break
    
    if filepath:
        print(f"Found balls.bin at: {filepath}\n")
        import_balls_geonodes(filepath)
    else:
        print(f"Error: Could not find balls.bin")
        print(f"Looked in:")
        for path in possible_paths:
            print(f"  - {path}")
        print("\nTo use manually:")
        print("  import_balls_geonodes('/path/to/balls.bin')")
