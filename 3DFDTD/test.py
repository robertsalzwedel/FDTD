from modules.fundamentals import Dimensions
import numpy as np
from modules.classes import Field

dim = 3
dims = Dimensions(x=dim, y=dim, z=dim)
a = Field(dims, 1)
b = Field(dims, 2)

print(a)
