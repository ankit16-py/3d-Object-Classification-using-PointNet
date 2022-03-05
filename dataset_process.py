import torch
import numpy as np
import os
import open3d as o3
import cv2

class ProcessDataset(torch.utils.data.Dataset):
    """
    This preprocesses the dataset.
    """

    def __init__(self, root, npoints=2500, classification=False, class_choice=None, train=True, image=False, seg_data=False):

        self.npoints = npoints
        self.seg_data= seg_data
        self.root = root
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.cat = {}
        self.image = image
        self.classification = classification

        # Map categories to folders in dataset
        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]

        if not class_choice is None:
            self.cat = {k: v for k, v in self.cat.items() if k in class_choice}

        # Build the data of points
        self.meta = {}
        for item in self.cat:
            self.meta[item] = []
            dir_point = os.path.join(self.root, self.cat[item], 'points')
            dir_seg = os.path.join(self.root, self.cat[item], 'points_label')
            dir_seg_img = os.path.join(self.root, self.cat[item], 'seg_img')

            fns = sorted(os.listdir(dir_point))
            if train:
                fns = fns[:int(len(fns) * 0.9)]
            else:
                fns = fns[int(len(fns) * 0.9):]

            for fn in fns:
                token = (os.path.splitext(os.path.basename(fn))[0])
                self.meta[item].append((os.path.join(dir_point, token + '.pts'), os.path.join(dir_seg, token + '.seg'),
                                        os.path.join(dir_seg_img, token + '.png')))

        # Build a datapath with items, points, seg_points
        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn[0], fn[1], fn[2]))

        self.classes = dict(zip(sorted(self.cat), range(len(self.cat))))

        self.num_seg_classes = 0
        if not self.classification:  # Take the Segmentation Labels
            for i in range(len(self.datapath) // 50):
                l = len(np.unique(np.loadtxt(self.datapath[i][-2]).astype(np.uint8)))
                if l > self.num_seg_classes:
                    self.num_seg_classes = l

    def __getitem__(self, index):
        '''
        This will be used to pick a specific element from the dataset.
        self.datapath is the dataset.
        Each element is under format "class, points, segmentation labels, segmentation image"
        '''
        # Get one Element
        fn = self.datapath[index]

        # get its Class
        cls = self.classes[fn[0]]

        # Read the Point Cloud
        point_set = np.asarray(o3.io.read_point_cloud(fn[1], format='xyz').points, dtype=np.float32)

        # Read the Segmentation Data
        seg = np.loadtxt(fn[2]).astype(np.int64)

        # Read the Segmentation Image
        image = cv2.imread(fn[3])

        choice = np.random.choice(len(seg), self.npoints, replace=True)

        # resample
        point_set = point_set[choice, :]
        seg = seg[choice]
        point_set = torch.from_numpy(point_set)
        seg = torch.from_numpy(seg)
        cls = torch.from_numpy(np.array([cls]).astype(np.int64))

        if self.classification:
            if self.image:
                if self.seg_data:
                    return point_set, cls, image, seg
                else:
                    return point_set, cls, image
            else:
                if self.seg_data:
                    return point_set, cls, seg
                else:
                    return point_set, cls

        else:
            if self.image:
                if self.seg_data:
                    return point_set, image, seg
                else:
                    return point_set, image
            else:
                if self.seg_data:
                    return point_set, seg
                else:
                    return point_set

    def __len__(self):
        return len(self.datapath)