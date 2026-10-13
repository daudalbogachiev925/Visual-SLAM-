import numpy as np

class Map3D:
    def __init__(self):
        self.points = []  # 3D точки
        self.descriptors = []

    def add(self, pts3d, desc):
        self.points.append(pts3d)
        self.descriptors.append(desc)

    def all_points(self):
        if not self.points:
            return np.zeros((0, 3))
        return np.vstack(self.points)

    def size(self):
        return len(self.points)
