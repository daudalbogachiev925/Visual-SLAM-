import numpy as np
from src.features import ORBExtractor

def test_orb_detects():
    img = (np.random.rand(480, 640) * 255).astype(np.uint8)
    ex = ORBExtractor()
    kp, des = ex.detect(img)
    assert des is not None
