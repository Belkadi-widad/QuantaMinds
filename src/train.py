# from
#  https://github.com/sfinos316/Methane-Plume-Segmentation/tree/main 
# under GNU General Public License v3.0
#
# Copyright (C) 2017 https://github.com/sfinos316 

# we also modified it to suit us.. 

import os
import numpy as np
import torch
from torch import nn, optim
from tqdm.autonotebook import tqdm
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, random_split, RandomSampler
from torch.utils.tensorboard import SummaryWriter
import argparse
from torchmetrics import JaccardIndex, Accuracy
import time
import random

from classical_unet import *
from data import create_dataset


#Device definition
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)



def set_all_seeds(seed):
    """Set all the seeds to fix the same initial conditions for each training;
    :param seed: seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def train_model(model, epochs, opt, loss, bs, bands, lr, 
                path_savemodel, 
                train_size =None , test_size=None, 
                TrainImgfiles =None, 
                 ):
    """Wrapper function for model training.
    :param model: model instance
    :param epochs: (int) number of epochs to be trained
    :param opt: optimizer instance
    :param loss: loss function instance
    :param bs: (int) batch size"""
    
    # create the datasets
    PATH_D_TRAIN=os.getcwd() + "/data/DataTrain/input_tiles/"
    PATH_S_TRAIN=os.getcwd()+"/data/DataTrain/output_matrix/"
    PATH_D_TEST=os.getcwd()+"/data/DataTest/input_tiles/"
    PATH_S_TEST=os.getcwd()+"/data/DataTest/output_matrix/"
    PATH_RUNS =os.getcwd()+'/runs/'

    # writer = SummaryWriter(PATH_RUNS+ f"ep{epochs}_lr{lr}_bs{bs}_time{crt_time}_idd{idd}/")
    
    data_train = create_dataset(
        datadir=PATH_D_TRAIN,
        segdir=PATH_S_TRAIN,
        band=bands,
	apply_transforms=True, imgfiles = TrainImgfiles )

    data_val = create_dataset(
        datadir=PATH_D_TEST,
        segdir=PATH_S_TEST,
        band=bands,
        apply_transforms=False)
    
    # if train_size is None and test_size is None: 
    #     train_size = int(2*len(data_train)/3) 
    #     test_size=len(data_val)
    
    # train_sampler = RandomSampler(data_train, replacement=True,
    #                               num_samples= train_size)
    
    val_sampler = RandomSampler(data_val, replacement=True,
                                num_samples=test_size)

    # initialize data loaders
    train_dl = DataLoader(data_train, batch_size=bs,
                          pin_memory=True)
    val_dl = DataLoader(data_val, batch_size=bs,
                         pin_memory=True, sampler = val_sampler )

    
    #Identification of the training
    t = time.localtime()
    crt_time = time.strftime("%H:%M:%S", t).replace(':', '_')
    idd = random.randint(0, 100)
    print(f"{crt_time}, {idd}")

    # start training process

    model_metrics ={}
    for epoch in range(epochs):

        model.train()
        train_loss_total = 0
        train_ious = []
        train_acc_total = 0
        train_area = []
        print('epoch', epoch)

        for i, batch in enumerate(train_dl):
            x = batch['img'].float().to(device)
            y = batch['fpt'].float().to(device)
            print('batch ', i)
            output = model(x)

            #derive segmentation map from prediction
            output_bin = torch.round(torch.sigmoid(output))

            # derive IoU values
            jaccard = JaccardIndex(task = 'binary').to(device)
            z = jaccard(output_bin, y.unsqueeze(dim=1)).cpu()
            train_ious.append(z.detach().numpy())

            # derive image-wise accuracy for this batch
            acc = Accuracy(task = 'binary').to(device)
            a = acc(output_bin, y[:,None,:,:])
            train_acc_total += a

            # derive loss
            loss_epoch = loss(output, y.unsqueeze(dim=1))
            train_loss_total += loss_epoch
         
            # derive smoke areas
            area_pred = torch.sum(output_bin, dim = (1, 2, 3))
            area_true = torch.sum(y.unsqueeze(dim=1), dim=(1,2,3))

            #derive area accuracy
            area_dif = torch.sum(torch.square(torch.sub(area_pred, area_true))).cpu()
            train_area.append(area_dif.detach().numpy())

            # learning
            opt.zero_grad()
            loss_epoch.backward()
            opt.step()
            
            # logging
            # writer.add_scalar("Train/Loss", train_loss_total/(i+1), epoch)
            print('Train/Loss' , train_loss_total/(i+1))
            # writer.add_scalar("Train/Iou", np.average(train_ious), epoch)
            print('Train/Iou' , np.average(train_ious))
            # writer.add_scalar("Train/Acc", train_acc_total/(i+1), epoch)
            print('Train/Acc' , train_acc_total/(i+1))

            # writer.add_scalar('Train/Arearatio mean', np.average(train_area), epoch)
            # writer.add_scalar('Train/Arearatio std', np.std(train_area), epoch)
            # writer.add_scalar('Train/learning_rate', opt.param_groups[0]['lr'], epoch)

            torch.cuda.empty_cache()

        # start evaluation process 
        with torch.no_grad():
            val_loss_total = 0
            val_ious = []
            val_acc_total = 0
            val_area = []
          
            for j, batch in enumerate(val_dl):
                x = batch['img'].float().to(device)
                y = batch['fpt'].float().to(device)

                output = model(x)

                # derive loss
                loss_epoch = loss(output, y.unsqueeze(dim=1))
                val_loss_total += loss_epoch
                
                #derive segmentation map from prediction
                output_bin = torch.round(torch.sigmoid(output))

                # derive IoU values
                jaccard = JaccardIndex(task = 'binary').to(device)
                z = jaccard(output_bin, y.unsqueeze(dim=1)).cpu()
                val_ious.append(z.detach().numpy())

                # derive image-wise accuracy for this batch
                acc = Accuracy(task = 'binary').to(device)
                a = acc(output_bin, y[:,None,:,:])
                val_acc_total += a

                # derive smoke areas
                area_pred = torch.sum(output_bin, dim = (1, 2, 3))
                area_true = torch.sum(y.unsqueeze(dim=1), dim=(1,2,3))

                #derive area accuracy
                area_dif = torch.sum(torch.square(torch.sub(area_pred, area_true))).cpu()
                val_area.append(area_dif.detach().numpy())
              
                # logging
                print('epoch: ', epoch)
                # writer.add_scalar("Test/Loss", val_loss_total/(j+1), epoch)
                print('Test Loss:' , val_loss_total/(j+1))
                # writer.add_scalar("Test/Iou", np.average(val_ious), epoch)
                print('Test Iou:' , np.average(val_ious))

                # writer.add_scalar("Test/Acc", val_acc_total/(j+1), epoch)
                print('Test Accuracy:' , val_acc_total/(j+1))

                # writer.add_scalar('Test/Arearatio mean',
                #                np.average(val_area), epoch)
                print('Test Arearatio mean:' , np.average(val_area))

                # writer.add_scalar('Test/Arearatio std',
                #                np.std(val_area), epoch)

                print('Test Arearatio std:' ,  np.std(val_area))


            #print the metrucs after each epoch
            print(("Epoch {:d}: train loss={:.3f}, val loss={:.3f}, " "train iou={:.3f}," "train acc={:.3f},").format(epoch+1, train_loss_total/(i+1), val_loss_total/(j+1), np.average(train_ious),train_acc_total/(i+1)))
            model_metrics[epoch+1] = {
                    "train_loss" : train_loss_total/(i+1), 
                    "val_loss" : val_loss_total/(j+1), 
                    "train_iou" : np.average(train_ious),
                    "train_acc" : train_acc_total/(i+1)
            }
            # writer.flush()
            model_name = f"{path_savemodel}\ep{epoch}_lr{lr}_time_{crt_time}_idd{idd}_val_loss{val_loss_total/(j+1)}_train_loss{train_loss_total/(i+1)}.model"

            #save the model each 50 epochs
            if (epoch+1)%5 == 0:
                # PATH_MOD=f"{os.getcwd()}\mod"
                # print(model.state_dict())
                torch.save(model.state_dict(), model_name)


    return model, model_metrics , model_metrics[epoch+1], model_name




################## MAIN ##################
# if __name__ == '__main__':

#     # setup argument parser
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-ep', type=int, default=300,
#                     help='Number of epochs')
#     parser.add_argument('-bs', type=int, nargs='?',
#                     default=60, help='Batch size')
#     parser.add_argument('-lr', type=float,
#                     nargs='?', default=0.01, help='Learning rate')
#     args = parser.parse_args()

#     print('heeellooo')
#     # set seed
#     set_all_seeds(21)
    
#     # model
#     bands = [1,2,3,4,5,6,7,8,9,10,11,12,13]
#     model = UNet(n_channels=13, n_classes=1)
#     model.to(device)
    
#     # trainning parameters
#     ep = args.ep 
#     lr = args.lr
#     bs = args.bs

#     # setup tensorboard writer
#     PATH_RUNS =os.getcwd()+'/runs/'
#     writer = SummaryWriter(PATH_RUNS+ f"ep{ep}_lr{lr}_bs{bs}_time{crt_time}_idd{idd}/")
    
#     # initialize loss function
#     loss = nn.BCEWithLogitsLoss()
    
#     # initialize optimizer
#     opt = optim.Adam(model.parameters(),lr=lr)
    
#     # create the directory to save the models
#     PATH_MOD=os.getcwd()+"/mod/"
#     if not os.path.exists(PATH_MOD):
#         os.mkdir(PATH_MOD)

#     # run training
#     trained_model = train_model(model, ep, opt, loss, bs)
    
#     # save final trained model
#     torch.save(trained_model.state_dict(),
#     PATH_MOD+f"ep{ep}_lr{lr}_bs{bs}_time{crt_time}_idd.model")    
    
#     writer.close()
