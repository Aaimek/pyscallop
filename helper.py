
from typing import Tuple


def distance(pt1: Tuple, pt2: Tuple):
    x, y = pt1
    a, b = pt2
    dist = (x-a)**2 + (y-b)**2
    
    return dist**0.5

