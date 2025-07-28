# triangle_on_sonar_finder

# Triangle Finder Module

This module provides a vision service that can detect triangles on a sonar screen.

## Configuration

Here's an example configuration:
```json
edit this and change the path:
{
  "camera_name": "camera-1",
  "path_to_templates_directory": "/path/to/templates",
  "threshold": 0.65,
  "scale (optional)": 0.5
}
```
Default downscale factor: 0.5

Warning: scaling down by more than 0.5 can affect detection accuracy. When scaling down by higher factors, increasing the threshold can help recover accuracy. 



