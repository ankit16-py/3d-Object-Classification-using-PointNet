# 3d-Object-Classification-using-PointNet
This project highlights the use of deep learning for 3d point cloud classification of common objects using Standford's ShapeNet dataset. 

<img src="deliv/output.gif" width= 900px>

## Packages Used
1) Pytorch
2) Numpy
3) Open3d
4) OpenCV 4

## How to use

1) Download the ShapeNet datatset from the official website and add the main dataset folder to root directory
2) Every python file has a root variable inside it. Change it to match the filename of the dataset folder
3) run the visualize_data.py file to visualize one random point cloud and it's corresponding segmentation image
4) To train the model run the model_train.py file. Change the outf variable to the desired folder name to store the trained models and create the corresponding folder. Change the corresponding data_root folder to match the downloaded dataset. The Open3d visualizations can be exited using the 'q' key. One can also change other hyperparameters like batch size and epoch number.
5) To perform inference use the model_eval.py file. The script will display random point cloud objects from the test datatset and the terminal prints their classification result. Change the MODEL_PATH and root variables to match the saved trained model file path and dataset path respectively. One can also change the MAX_SAMPLES variable to determine the number of inference runs.

