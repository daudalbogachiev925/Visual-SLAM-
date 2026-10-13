import cv2
import numpy as np
from src.features import ORBExtractor
from src.map import Map3D

class VisualSLAM:
    def __init__(self, K):
        self.K = K
        self.extractor = ORBExtractor()
        self.map = Map3D()
        self.prev_gray = None
        self.prev_kp = None
        self.prev_des = None
        self.R = np.eye(3)
        self.t = np.zeros((3, 1))
        self.trajectory = []

    def process(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp, des = self.extractor.detect(gray)

        if self.prev_des is not None:
            matches = self.extractor.match(self.prev_des, des)
            if len(matches) > 20:
                pts1 = np.float32([self.prev_kp[m.queryIdx].pt for m in matches])
                pts2 = np.float32([kp[m.trainIdx].pt for m in matches])
                E, mask = cv2.findEssentialMat(pts1, pts2, self.K, method=cv2.RANSAC)
                if E is not None:
                    _, R, t, mask = cv2.recoverPose(E, pts1, pts2, self.K)
                    self.t = self.t + self.R @ t
                    self.R = R @ self.R
                    self.trajectory.append(self.t.flatten().copy())

        self.prev_gray = gray
        self.prev_kp = kp
        self.prev_des = des
        return self.t.flatten()
