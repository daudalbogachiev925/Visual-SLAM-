import cv2
import numpy as np

class ORBExtractor:
    def __init__(self, n_features=1500):
        self.orb = cv2.ORB_create(nfeatures=n_features)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def detect(self, gray):
        kp, des = self.orb.detectAndCompute(gray, None)
        return kp, des

    def match(self, des1, des2, ratio=0.75):
        if des1 is None or des2 is None:
            return []
        matches = self.matcher.match(des1, des2)
        matches = sorted(matches, key=lambda m: m.distance)
        return matches[:200]
