import cv2
import numpy as np

def draw_trajectory(traj, size=600):
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    if len(traj) < 2:
        return canvas
    pts = np.array(traj)
    x = pts[:, 0]; z = pts[:, 2]
    x = (x - x.min()) / (x.ptp() + 1e-6) * (size - 100) + 50
    z = (z - z.min()) / (z.ptp() + 1e-6) * (size - 100) + 50
    for i in range(1, len(x)):
        cv2.line(canvas, (int(x[i-1]), int(z[i-1])), (int(x[i]), int(z[i])),
                 (0, 255, 0), 2)
    return canvas
