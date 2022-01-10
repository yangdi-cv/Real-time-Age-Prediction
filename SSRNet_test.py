from SSRNet_dataloader import AgeDataset as AD
from torch.utils.data import DataLoader as DataLoader
import torch
from torch.autograd import Variable
import torch.nn as nn
from network.SSR_Network import SSRNet
import math
from tqdm import tqdm
import time

def test(model, criterion, dataloader):

    # Load the model
    model.load_state_dict(torch.load(model_file))
    model.eval()

    mae, mse, num_imgs, test_loss = 0, 0, 0, 0

    with torch.no_grad():
        for img, label in tqdm(dataloader):
            img, label = Variable(img).cuda(), Variable(label).cuda()
            out = model(img)
            t_loss = criterion(out, label.squeeze())
            test_loss += t_loss.item()

            # Calculate the current performance
            num_imgs += label.squeeze().size(0)
            mae += torch.sum(torch.abs(out - label.squeeze()))
            mse += torch.sum((out - label.squeeze()) ** 2)

        test_LOSS = test_loss / len(dataloader)
        MAE = mae.float() / num_imgs
        MSE = mse.float() / num_imgs
        RMSE = math.sqrt(mse.float() / num_imgs)

    return test_LOSS, MAE, MSE, RMSE

if __name__ == '__main__':

    dataset_dir = './data/'
    model_file = './model/SSRNet/model_SSRNet_batch20_epoch31.pth'

    workers = 10
    batch_size = 20

    # Set the model
    model = SSRNet(image_size=64)
    model.cuda()
    model = nn.DataParallel(model)

    # Define the loss function
    criterion = nn.L1Loss()  # MAE loss

    # Load data
    datafile = AD('test', dataset_dir)
    dataloader = DataLoader(datafile, batch_size=batch_size,
                            shuffle=True, num_workers=workers)
    print('Dataset loaded! length of test set is {0}'
          .format(len(datafile)))
    T1 = time.time()

    # Test model
    test_LOSS, MAE, MSE, RMSE= test(model, criterion, dataloader)
    print('Test loss:{:.4f} | MAE:{:.4f} | MSE:{:.4f} | RMSE:{:.4f}\n'
          .format(test_LOSS, MAE, MSE, RMSE))
    T2 = time.time()
    print('Test time:', T2 - T1, 's')
