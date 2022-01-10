from SSRNet_dataloader import AgeDataset as AD
import torch
from torch.utils.data import DataLoader as DataLoader
from torch.autograd import Variable
import torch.nn as nn
from network.SSR_Network import SSRNet
import math
from tqdm import tqdm
import time

def train(model, optimizer, criterion, train_loader, cnt):
    model.train()

    mae, mse, num_imgs, training_loss = 0, 0, 0, 0

    for img, label in tqdm(train_loader):
        img, label = Variable(img).cuda(), Variable(label).cuda()

        # Perform a feed-forward pass
        out = model(img)

        # Compute the batch loss
        t_loss = criterion(out, label.squeeze())
        training_loss += t_loss.item()

        # Calculate the current performance
        num_imgs += label.squeeze().size(0)
        mae += torch.sum(torch.abs(out - label.squeeze()))
        mse += torch.sum((out - label.squeeze()) ** 2)

        # Set the gradients to zero before starting backpropagation
        optimizer.zero_grad()

        # Compute gradient
        t_loss.backward()

        # Updates the trainable weights
        optimizer.step()

    train_Loss = training_loss / len(train_loader)
    train_mae = mae.float() / num_imgs
    train_mse = mse.float() / num_imgs
    train_rmse = math.sqrt(mse.float() / num_imgs)


    return train_Loss, train_mae, train_mse, train_rmse


def validate(model, criterion, val_loader, cnt):
    model.eval()

    mae, mse, num_imgs, validation_loss = 0, 0, 0, 0

    with torch.no_grad():
        for img, label in tqdm(val_loader):
            img, label = Variable(img).cuda(), Variable(label).cuda()

            # Perform a forward pass
            out = model(img)

            # Calculate the validation loss
            v_loss = criterion(out, label.squeeze())
            validation_loss += v_loss.item()

            # Calculate the current performance
            num_imgs += label.squeeze().size(0)
            mae += torch.sum(torch.abs(out - label.squeeze()))
            mse += torch.sum((out - label.squeeze()) ** 2)

        val_Loss = validation_loss / len(val_loader)
        val_mae = mae.float() / num_imgs
        val_mse = mse.float() / num_imgs
        val_rmse = math.sqrt(mse.float() / num_imgs)


        return val_Loss, val_mae, val_mse, val_rmse

if __name__ == '__main__':

    dataset_dir = './data/'
    model_cp = './model/'
    workers = 10
    batch_size = 20
    lr = 0.0001
    nepoch = 50


    # Create DataSet objects for training and validation
    train_data = AD('train', dataset_dir)
    train_loader = DataLoader(train_data, batch_size=batch_size,
                              shuffle=True, num_workers=workers)

    val_data = AD('validate', dataset_dir)
    val_loader = DataLoader(val_data, batch_size=batch_size,
                            shuffle=True, num_workers=workers)

    print('Dataset loaded!\tLength of train set is {0}\tLength of validate set is {1}'
          .format(len(train_data), len(val_data)))

    model = SSRNet(image_size=64)
    model = model.cuda()
    model = nn.DataParallel(model)

    # Define the loss function and optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.L1Loss()  # MAE loss

    # Start training the model
    cnt = 0
    best_MAE = 999
    patience = 50

    T1 = time.time()

    for epoch in range(nepoch):
        # Train the model over a single epoch
        train_loss, train_mae, train_mse, train_rmse = train(model, optimizer, criterion, train_loader, cnt)

        # Validate the model
        val_loss, val_mae, val_mse, val_rmse= validate(model, criterion, val_loader, cnt)

        print('-----------------------------------------------------\n'
              'Epoch: {}\n'
              'Training MAE: {:.4f} | Training MSE: {:.4f} | Training RMSE: {:.4f}\n'
              'Valid MAE: {:.4f} | Valid MSE: {:.4f} | Valid RMSE: {:.4f}\n'
              '-----------------------------------------------------'
              .format(epoch, train_mae, train_mse,train_rmse, val_mae, val_mse, val_rmse))

        cnt = cnt + 1

        if val_mae < best_MAE:
            print('------------------------------------------------------\n'
                  'Save model: Valid MAE decreased from {:.4f} to {:.4f}.\n'
                  '------------------------------------------------------'
                  .format(best_MAE, val_mae))

            torch.save(model.state_dict(),
                       './model/SSRNet/model_SSRNet_batch{}_epoch{}.pth'.format(batch_size, epoch))

            best_MAE = val_mae
            not_improved_count = 0  # reset count

            T2 = time.time()
            print('Training time:',T2 - T1,'s')
        else:
            not_improved_count += 1  # increment count

        if not_improved_count >= patience:
            T3 = time.time()
            break

    print('Training total time:', T3 - T1, 's')