from point_net_model import PointNet
import torch
from dataset_process import ProcessDataset
from random import randrange
import open3d as o3
from torch.autograd import Variable
import numpy as np

def read_pointnet_colors(seg_labels):
    map_label_to_rgb = {
        1: [255, 255, 0],
        2: [0, 0, 255],
        3: [255, 0, 0],
        4: [0, 255, 255],
        5: [255, 0, 255],
        6: [0, 255, 0],
    }
    colors = np.array([map_label_to_rgb[label] for label in seg_labels])
    return colors

NUM_POINTS = 10000
root= 'shapenetcore_partanno_segmentation_benchmark_v0'
MODEL_PATH = 'model_checkpoint/model@_epoch_2.pth'

classes_dict = {'Airplane': 0, 'Bag': 1, 'Cap': 2, 'Car': 3, 'Chair': 4,
                'Earphone': 5, 'Guitar': 6, 'Knife': 7, 'Lamp': 8, 'Laptop': 9,
                'Motorbike': 10, 'Mug': 11, 'Pistol': 12, 'Rocket': 13,
                'Skateboard': 14, 'Table': 15}
num_classes = len(classes_dict.items())

# Create the classification network from pre-trained model
model = PointNet(k=num_classes, num_points=NUM_POINTS)
if torch.cuda.is_available():
    model.cuda()
    model.load_state_dict(torch.load(MODEL_PATH))
else:
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
model.eval()

test_dataset = ProcessDataset(root=root, train=False, classification=False, npoints=NUM_POINTS, seg_data=True)

MAX_SAMPLES = 5

for samples in range(MAX_SAMPLES):
    random_index = randrange(len(test_dataset))
    print('[Sample {} / {}]'.format(random_index, len(test_dataset)))

    # get next sample
    point_set, seg = test_dataset.__getitem__(random_index)

    # create cloud for visualization
    cloud = o3.geometry.PointCloud()
    cloud.points = o3.utility.Vector3dVector(point_set)
    cloud.colors = o3.utility.Vector3dVector(read_pointnet_colors(seg.numpy()))

    # perform inference in GPU
    points = Variable(point_set.unsqueeze(0))
    points = points.transpose(2, 1)
    if torch.cuda.is_available():
        points = points.cuda()
    pred_logsoft, _ = model(points)

    # move data back to cpu for visualization
    pred_logsoft_cpu = pred_logsoft.data.cpu().numpy().squeeze()
    pred_soft_cpu = np.exp(pred_logsoft_cpu)
    pred_class = np.argmax(pred_soft_cpu)

    o3.visualization.draw_geometries([cloud])

    input('Your object is a [{}] with probability {:0.3}. Press enter to continue!'
          .format(list(classes_dict.keys())[pred_class], pred_soft_cpu[pred_class]))