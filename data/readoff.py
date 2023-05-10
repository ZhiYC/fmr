import igl
import numpy as np

data = igl.read_triangle_mesh('ModelNet40/airplane/train/airplane_0001.off')
igl.write_triangle_mesh('test.off',data[0],data[1])
print(data[0].shape)