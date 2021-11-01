import os.path as osp

import mmcv
import numpy as np
from PIL import Image

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class DOTADataset(CustomDataset):
    '''
    First, you should transform DOTA format to yolo format or not,maybe not
    '''
    CLASSES = ('plane', 'baseball-diamond',
                'bridge', 'ground-track-field',
                'small-vehicle', 'large-vehicle',
                'ship', 'tennis-court',
                'basketball-court', 'storage-tank',
                'soccer-ball-field', 'roundabout',
                'harbor', 'swimming-pool',
                'helicopter', 'container-crane')

    def __init__(self,img_subdir='images',ann_subdir='split_train/labelTxt',**kwargs):
        super(DOTADataset, self).__init__(**kwargs)
        self.img_subdir=img_subdir
        self.ann_subdir=ann_subdir
        self.cat2label = {cat: i for i, cat in enumerate(self.CLASSES)}

    def load_annotations(self, ann_file):
        data_infos = []
        img_paths = mmcv.list_from_file(ann_file)
        for img_id,img_path in enumerate(img_paths):
            # img_path = osp.join(self.img_prefix,self.img_subdir, f'{img_id}.jpg')
            filename=img_path.split('/')[-1] #.split('.')[0]
            img = Image.open(img_path)
            width, height = img.size
            data_infos.append(
                dict(id=img_id, filename=filename, width=width, height=height))
        return data_infos

    def get_ann_info(self, idx):
        # img_id=self.data_infos[idx]['id']
        img_name=self.data_infos[idx]['filename'].split('.')[0]
        txt_path = osp.join(self.data_root, self.ann_subdir, f'{img_name}.txt')
        polygons = []
        labels = []
        difficult=[]
        ann_list = mmcv.list_from_file(txt_path)
        for ann in ann_list:
            ann=ann.split(' ')
            polygons.append([float(coord) for coord in ann[:8]])
            labels.append(int(self.cat2label[ann[8]]))
            difficult.append(int(ann[9]))
        ann = dict(
            polygons= np.array(polygons, dtype=np.float32),
            labels=np.array(labels, dtype=np.int64),
            difficult=np.array(difficult, dtype=np.int64))
        return ann