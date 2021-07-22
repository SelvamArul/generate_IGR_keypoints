import torch
from point import Point3D


p1 = Point3D([0, 0, 0])

p2 = Point3D([0, 0, 1])

length = p1 - p2


print (3 * p2)

t = torch.linspace(0, 1, steps=4).tolist()
t = t[1:-1] # remove start and end points


print ("length ", length)

new_points = [ p1 + (x * length) for x in t ]

print ("new_points ", new_points)


min_points = Point3D([-1, -1, -1])
max_points = Point3D([ 1,  1,  1])

bbox_points = {
    'A': Point3D([min_points.x, min_points.y, min_points.z]),
    'B': Point3D([min_points.x, min_points.y, max_points.z]),

    'C': Point3D([min_points.x, max_points.y, min_points.z]),
    'D': Point3D([min_points.x, max_points.y, max_points.z]),

    'E': Point3D([max_points.x, min_points.y, min_points.z]),
    'F': Point3D([max_points.x, min_points.y, max_points.z]),

    'G': Point3D([max_points.x, max_points.y, min_points.z]),
    'H': Point3D([max_points.x, max_points.y, max_points.z]),
}

keypoints = [v for (k,v) in  bbox_points.items()]

print ("keypoints\n", keypoints)

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
    print (p1, p2)
    print ("new_points \n", new_points)
    print ()
    print ()
    keypoints = [*keypoints, *new_points]

print (keypoints)