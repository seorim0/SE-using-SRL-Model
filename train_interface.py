"""
Train interface for speech enhancement!
You can just run this file.
"""
import os
import argparse
import torch
import options
import utils
import datetime
import random
import numpy as np
import time
from dataloader import create_dataloader


######################################################################################################################
#                                                  Parser init                                                       #
######################################################################################################################
opt = options.Options().init(argparse.ArgumentParser(description='speech enhancement')).parse_args()
print(opt)

######################################################################################################################
#                                    Set a model (check point) and a log folder                                      #
######################################################################################################################
if not opt.pretrained or opt.pretrained_init:
    dir_name = os.path.dirname(os.path.abspath(__file__))  # absolute path
    print(dir_name)

    date = datetime.datetime.now()
    log_dir = os.path.join(dir_name, 'log', opt.arch + '_' + str(date.month) + str(date.day) + '_' + opt.env)
    utils.mkdir(log_dir)
    print("Now time is : ", date.isoformat())
    tboard_dir = os.path.join(log_dir, 'logs')
    model_dir = os.path.join(log_dir, 'models')
    utils.mkdir(model_dir)  # make a dir if there is no dir (given path)
    utils.mkdir(tboard_dir)
else:
    model_dir = opt.pretrain_model_path[:-17] + 'models'
    tboard_dir = opt.pretrain_model_path[:-17] + 'logs'

######################################################################################################################
#                                                   Model init                                                       #
######################################################################################################################
# set device
DEVICE = torch.device(opt.device)

# set seeds
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

# define model
model = utils.get_arch(opt)

total_params = utils.cal_total_params(model)
print('total params : %d (%.2f M, %.2f MBytes)\n' %
      (total_params,
       total_params / 1000000.0,
       total_params * 4.0 / 1000000.0))

# define loss type
trainer, validator = utils.get_train_mode(opt)
loss_calculator = utils.get_loss(opt)

# define optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr_initial)
model = model.to(DEVICE)

if opt.stage1_training:
    from s3prl.nn import S3PRLUpstream, Featurizer
    srl_model = S3PRLUpstream("wavlm_large").to(DEVICE)
    featurizer = Featurizer(srl_model)
    featurizer = featurizer.to(DEVICE)


# load the params if there is pretrained model
epoch_start_idx = 1
if opt.pretrained:
    print('Load the pretrained model...')
    chkpt = torch.load(opt.pretrain_model_path, map_location=DEVICE)
    model.load_state_dict(chkpt['model'], strict=False)
    if opt.stage1_training:
        featurizer.load_state_dict(chkpt['featurizer'], strict=False)

    if not opt.pretrained_init:
            optimizer.load_state_dict(chkpt['optimizer'])
            epoch_start_idx = chkpt['epoch'] + 1

            utils.optimizer_to(optimizer, DEVICE)


scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.decay_epoch, gamma=0.5)

######################################################################################################################
#                                               Create Dataloader                                                    #
######################################################################################################################
train_loader = create_dataloader(opt, mode='train')
valid_loader = create_dataloader(opt, mode='valid')
print("Sizeof training set: ", train_loader.__len__(),
      ", sizeof validation set: ", valid_loader.__len__())

######################################################################################################################
######################################################################################################################
#                                             Main program - train                                                   #
######################################################################################################################
######################################################################################################################
writer = utils.Writer(tboard_dir)
train_log_fp = open(model_dir + '/train_log.txt', 'a')

print('Train start...')
if opt.stage1_training:
    for epoch in range(epoch_start_idx, opt.nepoch + 1):
        st_time = time.time()

        # train
        train_loss = trainer(model, srl_model, featurizer, train_loader, loss_calculator, optimizer,
                             writer, epoch, DEVICE, opt)

        # save checkpoint file to resume training
        save_path = str(model_dir + '/chkpt_%d.pt' % epoch)
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'featurizer': featurizer.state_dict(),
            'epoch': epoch
        }, save_path)

        # validate
        valid_loss, pesq, stoi = validator(model, srl_model, featurizer, valid_loader, loss_calculator,
                                           writer, epoch, DEVICE, opt)

        print('EPOCH[{}] T {:.6f} |  V {:.6f}  takes {:.3f} seconds'
              .format(epoch, train_loss, valid_loss, time.time() - st_time))
        print('PESQ {:.6f} | STOI {:.6f}'.format(pesq, stoi))

        # write train log
        train_log_fp.write('EPOCH[{}] T {:.6f} |  V {:.6f}  takes {:.3f} seconds'
                           .format(epoch, train_loss, valid_loss, time.time() - st_time))
        train_log_fp.write('PESQ {:.6f} | STOI {:.6f}\n'.format(pesq, stoi))

        # scheduler
        scheduler.step()
else:
    for epoch in range(epoch_start_idx, opt.nepoch + 1):
        st_time = time.time()

        # train
        train_loss = trainer(model, train_loader, loss_calculator, optimizer,
                             writer, epoch, DEVICE, opt)

        # save checkpoint file to resume training
        save_path = str(model_dir + '/chkpt_%d.pt' % epoch)
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch
        }, save_path)

        # validate
        valid_loss, pesq, stoi = validator(model, valid_loader, loss_calculator,
                                           writer, epoch, DEVICE, opt)

        print('EPOCH[{}] T {:.6f} |  V {:.6f}  takes {:.3f} seconds'
              .format(epoch, train_loss, valid_loss, time.time() - st_time))
        print('PESQ {:.6f} | STOI {:.6f}'.format(pesq, stoi))

        # write train log
        train_log_fp.write('EPOCH[{}] T {:.6f} |  V {:.6f}  takes {:.3f} seconds'
                           .format(epoch, train_loss, valid_loss, time.time() - st_time))
        train_log_fp.write('PESQ {:.6f} | STOI {:.6f}\n'.format(pesq, stoi))

        # scheduler
        scheduler.step()

print('Training has been finished.')
train_log_fp.close()
