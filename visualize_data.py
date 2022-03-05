from dataset_process import ProcessDataset
import numpy as np
import cv2
import open3d as o3

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

root= 'shapenetcore_partanno_segmentation_benchmark_v0'

classes_dict = {'Airplane': 0, 'Bag': 1, 'Cap': 2, 'Car': 3, 'Chair': 4,
                'Earphone': 5, 'Guitar': 6, 'Knife': 7, 'Lamp': 8, 'Laptop': 9,
                'Motorbike': 10, 'Mug': 11, 'Pistol': 12, 'Rocket': 13,
                'Skateboard': 14, 'Table': 15}

data_obj= ProcessDataset(root=root, classification=True, image=True, seg_data=True)

idx= np.random.randint(0, len(data_obj))

pts, cls, img, seg= data_obj[idx]

classes_dict_list = list(classes_dict)

print(type(img))

print("The class of this point-cloud is: ", classes_dict_list[cls.item()])

display_cloud= o3.geometry.PointCloud()

display_cloud.points = o3.utility.Vector3dVector(pts)
display_cloud.colors = o3.utility.Vector3dVector(read_pointnet_colors(seg.numpy()))
o3.visualization.draw_geometries([display_cloud])
cv2.imshow("Segmented Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

