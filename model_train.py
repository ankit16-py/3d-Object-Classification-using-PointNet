from dataset_process import ProcessDataset
from point_net_model import PointNet
import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

batchSize = 32
num_points = 2500
workers = 1
epochs = 3
outf="model_checkpoint"
data_root= 'shapenetcore_partanno_segmentation_benchmark_v0'
classes_dict = {'Airplane': 0, 'Bag': 1, 'Cap': 2, 'Car': 3, 'Chair': 4,
                'Earphone': 5, 'Guitar': 6, 'Knife': 7, 'Lamp': 8, 'Laptop': 9,
                'Motorbike': 10, 'Mug': 11, 'Pistol': 12, 'Rocket': 13,
                'Skateboard': 14, 'Table': 15}

train_dataset = ProcessDataset(root = data_root, classification = True, npoints = num_points)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batchSize, shuffle=True, num_workers=workers)
test_dataset = ProcessDataset(root = data_root, classification = True, npoints = num_points, train=False)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batchSize, shuffle=True, num_workers=workers)

num_classes = len(classes_dict.items())
model = PointNet(k = num_classes, num_points = num_points)

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
if torch.cuda.is_available():
    model.cuda()

num_batch = len(train_dataset)/batchSize

for epoch in range(epochs):
    for i, data in enumerate(train_dataloader, 0):

        points, target = data
        points, target = Variable(points), Variable(target[:, 0])
        points = points.transpose(2, 1)
        if torch.cuda.is_available():
            points, target = points.cuda(), target.cuda()

        optimizer.zero_grad()
        classifier = model.train()
        pred, _ = model(points)
        loss = F.nll_loss(pred, target)
        loss.backward()
        optimizer.step()
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.data).cpu().sum()

        if i % 50 == 0:
            print("Metrics After Batch {}".format(i))
            print('[%d: %d/%d] train loss: %f train accuracy: %f' % (
                epoch, i, num_batch, loss.item(), correct.item() / float(batchSize)))

            j, data = next(enumerate(test_dataloader, 0))
            points, target = data
            points, target = Variable(points), Variable(target[:, 0])
            points = points.transpose(2, 1)
            if torch.cuda.is_available():
                points, target = points.cuda(), target.cuda()
            classifier = model.eval()
            pred, _ = model(points)
            loss = F.nll_loss(pred, target)
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.data).cpu().sum()
            print('[%d: %d/%d] val loss: %f val accuracy: %f\n' % (
            epoch, i, num_batch, loss.item(), correct.item() / float(batchSize)))

    torch.save(model.state_dict(), '%s/model@_epoch_%d.pth' % (outf, epoch))