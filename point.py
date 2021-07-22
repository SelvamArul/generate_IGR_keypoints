import torch
import numpy as np

class Point3D:
    def __init__(self, _p):
        if isinstance(_p, list):
            if len(_p) != 3:
                raise ValueError("Expected point to be 3 dimensional vector")
            self.point = torch.Tensor(_p)
        elif isinstance(_p, (np.ndarray, torch.Tensor)):
            p = _p.squeeze()
            if p.shape[0] != 3:
                raise ValueError("Expected point to be 3 dimensional vector")
            if isinstance(p, np.ndarray):
                self.point = torch.from_numpy(p)
            else:
                self.point = p
        else:
            raise ValueError("Unknown type")
        
        self.point.requires_grad = False
    
    @property
    def x(self):
        return self.point[0]

    @x.setter
    def x(self, _x):
        self.point[0] = _x

    @property
    def y(self):
        return self.point[1]

    @y.setter
    def y(self, _y):
        self.point[1] = _y

    @property
    def z(self):
        return self.point[2]

    @z.setter
    def z(self, _z):
        self.point[2] = _z
    
    def __repr__(self):
        return  str(self.point)

    def to_tensor(self):
        return self.point
    
    def __add__(self, p1):
        if not isinstance(p1, Point3D):
            raise ValueError("ValueError: Expected Point3D")
        return Point3D(self.to_tensor() + p1.to_tensor())
    
    def __sub__(self, p1):
        if not isinstance(p1, Point3D):
            raise ValueError("ValueError: Expected Point3D")
        return Point3D(self.to_tensor() - p1.to_tensor())
    
    def __mul__(self, a):
        if not isinstance(a, (int, float)):
            raise ValueError("ValueError: Expected (int, float)")
        return Point3D(self.to_tensor() * a)
    
    def __rmul__(self, a):
        if not isinstance(a, (int, float)):
            raise ValueError("ValueError: Expected (int, float)")
        return Point3D(self.to_tensor() * a)
    def abs(self):
        return Point3D(self.to_tensor().abs())