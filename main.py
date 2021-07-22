import torch
import stillleben as sl
from point import Point3D

from pathlib import Path as P

MODELS_DIR = P('/home/user/periyasa/workspace/ycb_models/bop_format')



def generate_keypoints(bbox_min, bbox_max, num_interpolate=2):
    t = torch.linspace(0, 1, steps=num_interpolate+2).tolist()
    t = t[1:-1] # remove start and end points

    bbox_points = {
    'A': Point3D([bbox_min.x, bbox_min.y, bbox_min.z]),
    'B': Point3D([bbox_min.x, bbox_min.y, bbox_max.z]),

    'C': Point3D([bbox_min.x, bbox_max.y, bbox_min.z]),
    'D': Point3D([bbox_min.x, bbox_max.y, bbox_max.z]),

    'E': Point3D([bbox_max.x, bbox_min.y, bbox_min.z]),
    'F': Point3D([bbox_max.x, bbox_min.y, bbox_max.z]),

    'G': Point3D([bbox_max.x, bbox_max.y, bbox_min.z]),
    'H': Point3D([bbox_max.x, bbox_max.y, bbox_max.z]),
    }
    
    keypoints = [v for (k,v) in  bbox_points.items()]
    pairs = [
    ['A', 'B'],
    ['A', 'C'],
    ['A', 'E'],
    ['B', 'D'],
    ['B', 'F'],
    ['C', 'D'],
    ['C', 'G'],
    ['D', 'H'],
    ['E', 'F'],
    ['E', 'G'],
    ['F', 'H'],
    ['G', 'H']
    ]

    for pair in pairs:
        _p1, _p2 = pair
        
        p1 = bbox_points[_p1]
        p2 = bbox_points[_p2]
        length = (p1 - p2).abs()
        new_points = [ p1 + (x * length) for x in t ]
        keypoints = [*keypoints, *new_points]
    keypoints_torch = [k.to_tensor() for k in keypoints]

    keypoints_torch.append(torch.tensor([0., 0., 0.])) # origin at the end always
    keypoints_torch = torch.vstack(keypoints_torch)

    return keypoints_torch

if __name__ == '__main__':
    sl.init()

    for i in range(1, 22):
        
        mesh_file = MODELS_DIR / f'obj_{i:06d}.ply'
        print ('mesh_file ', mesh_file)
        
        mesh = sl.Mesh(mesh_file)

        # NOTE: BOP meshes are in mm.
        # standard units is m. and also sl uses m.
        torch.save(mesh.bbox.max * 0.001, f'{i:06d}_max.pt')
        torch.save(mesh.bbox.min * 0.001, f'{i:06d}_min.pt')
        torch.save(mesh.bbox.center * 0.001, f'{i:06d}_center.pt')

        bbox_min = Point3D(mesh.bbox.min * 0.001)
        bbox_max = Point3D(mesh.bbox.max * 0.001)
        keypoints = generate_keypoints(bbox_min, bbox_max)
        torch.save(keypoints, f'{i:06d}_keypoints.pt')

        