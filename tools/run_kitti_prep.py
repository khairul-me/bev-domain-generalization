import sys
import os
sys.path.insert(0, 'E:/bev_research')
from config.paths import BEVFORMER_ROOT
os.chdir(BEVFORMER_ROOT)
sys.path.append('.')
from tools.create_data import kitti_data_prep
kitti_data_prep(root_path='C:/datasets/kitti', info_prefix='kitti', version='v1.0', out_dir='C:/datasets/kitti')
print('KITTI PREP DONE')
