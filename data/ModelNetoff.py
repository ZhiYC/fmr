import glob
import os
import numpy as np

def get_min_max(file_dir):
    min = 99999
    max = -99999
    for path in glob.iglob(os.path.join(file_dir,'test/*'),recursive=True):
        file_name = os.path.basename(path).split('.')[0]
        file_index = int(file_name.split('_')[-1])
        if file_index < min:
            min = file_index
        if file_index > max:
            max = file_index
    return min,max

if __name__ == '__main__':
    with open('ModelNet40.csv','w') as f:
        paths = glob.iglob('ModelNet40/*',recursive=True)
        paths = sorted(paths)
        for path in paths:
            subpath = path.split('/')[-1]
            f.write(f'{subpath},{get_min_max(path)[0]},{get_min_max(path)[1]}\n')