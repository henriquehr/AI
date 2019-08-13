import os
import time
import glob

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim

from PIL import Image
from skimage import io, transform

from data_loader import Rescale
from data_loader import RandomCrop
from data_loader import ToTensor
from data_loader import RandomHorizontalFlip
from data_loader import RandomVerticalFlip
from data_loader import DatasetLoader

from model import MYNet

import pytorch_ssim
import pytorch_iou

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


# bce_loss = nn.BCELoss(size_average=True)
bce_loss = torch.nn.BCELoss(reduction='mean')
# ssim_loss = pytorch_ssim.SSIM(window_size=11, size_average=True)
# iou_loss = pytorch_iou.IOU(size_average=True)


# def dice_loss(input, target):
#     smooth = 1.
#     iflat = input.view(-1)
#     tflat = target.view(-1)
#     intersection = (iflat * tflat).sum()
#     return 1 - ((2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))


def bce_ssim_loss(pred, target):
    # dice_out = dice_loss(pred, target)
    bce_out = bce_loss(pred, target)
    # ssim_out = 1 - ssim_loss(pred, target)
    # iou_out = iou_loss(pred, target)

    # loss = dice_out + ssim_out + iou_out
    loss = bce_out# + ssim_out + iou_out

    return loss


def muti_bce_loss_fusion(out, labels):
    out0, out1, out2, out3 = out
    loss0 = bce_ssim_loss(out0, labels)
    loss1 = bce_ssim_loss(out1, labels)
    loss2 = bce_ssim_loss(out2, labels)
    loss3 = bce_ssim_loss(out3, labels)
    # loss4 = bce_ssim_loss(out4, labels)

    loss = loss0 + loss1 + loss2 + loss3
 
    return loss, loss0, loss1, loss2, loss3


def normPRED(d):
	ma = torch.max(d)
	mi = torch.min(d)
	return (d-mi)/(ma-mi)


def save_output(image_name, predict, d_dir):
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()

    im = Image.fromarray(predict_np * 255).convert('L')#.convert('RGB')
    img_name = os.path.split(image_name)[-1]
    image = io.imread(image_name)
    imo = im.resize((image.shape[1], image.shape[0]), resample=Image.BILINEAR)

    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1,len(bbb)):
        imidx = imidx + "." + bbb[i]

    imo.save(d_dir + imidx + '.png')


def main():
    # data_dir = './train_data/'
    train_image_dir = './train_data/DUTS/DUTS-TR-Image/'
    train_label_dir = './train_data/DUTS/DUTS-TR-Mask/'

    model_dir = './saved_models/'

    resume_train = True
    saved_model_path = model_dir + 'model.pth'

    validation = True
    save_every = 1
    epoch_num = 100000
    batch_size_train = 16
    batch_size_val = 1
    train_num = 0
    val_num = 0

    if validation:
        val_image_dir = 'test_data/val/images/'
        val_label_dir = 'test_data/val/gts/'
        prediction_dir = './val_results/'

        val_img_name_list = glob.glob(val_image_dir + '*.jpg')
        val_lbl_name_list = glob.glob(val_label_dir + '*.png')

        val_dataset = DatasetLoader(img_name_list = val_img_name_list, 
            lbl_name_list = val_lbl_name_list,
            transform=transforms.Compose([
                Rescale(256),
                ToTensor()
            ]))
        
        val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)

    train_img_name_list = glob.glob(train_image_dir + '*.jpg')
    train_lbl_name_list = []

    for img_path in train_img_name_list:
        img_path = img_path.replace('.jpg', '.png')
        img_path = img_path.replace('DUTS-TR-Image', 'DUTS-TR-Mask')
        train_lbl_name_list.append(img_path)

    if len(train_img_name_list) == 0 or len(val_img_name_list) == 0:
        print('0 images found.')
        assert False

    print('Train images: ', len(train_img_name_list))
    print('Train labels: ', len(train_lbl_name_list))

    train_num = len(train_img_name_list)

    dataset = DatasetLoader(
        img_name_list=train_img_name_list,
        lbl_name_list=train_lbl_name_list,
        transform=transforms.Compose([
            RandomHorizontalFlip(0.5),
            RandomVerticalFlip(0.5),
            Rescale(300),
            RandomCrop(256),
            ToTensor()]))
    dataloader = DataLoader(dataset, batch_size=batch_size_train, shuffle=True, num_workers=4)

    model = MYNet(3, 1)
    model.cuda()

    from torchsummary import summary
    summary(model, input_size=(3, 256, 256))

    # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.00001, nesterov=False)
    optimizer = optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200000, 350000], gamma=0.1, last_epoch=-1)

    # scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.0001, 
    #     max_lr=0.01, step_size_up=8000, mode='triangular2')

    i_num_tot = 0
    loss_output = 0.0
    loss_pre_ref = 0.0
    i_num_epoch = 0
    epoch_init = 0

    if resume_train:
        print('Loading checkpoint: ', saved_model_path)
        checkpoint = torch.load(saved_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict']),
        epoch_init = checkpoint['epoch'] + 1
        i_num_tot = checkpoint['i_num_tot'] + 1
        i_num_epoch = checkpoint['i_num_epoch']
        loss_output = checkpoint['loss_output']
        # loss_pre_ref = checkpoint['loss_pre_ref']

    log_file = open('logs/log.txt', 'a+')
    log_file.write(str(model) + '\n')
    log_file.close()

    print('Training...')
    _s = time.time()
    for epoch in range(epoch_init, epoch_num):
        model.train()
        print('Epoch {}...'.format(epoch))
        _time_epoch = time.time()
        for i, data in enumerate(dataloader):
            i_num_tot += 1
            i_num_epoch += 1

            inputs, labels = data

            inputs = inputs.cuda()
            labels = labels.cuda()

            optimizer.zero_grad()

            out = model(inputs)
            loss = muti_bce_loss_fusion(out, labels)

            loss[0].backward()
            optimizer.step()
            scheduler.step()

            loss_output += loss[0].item()
            # loss_pre_ref += loss[1].item()

            del out, inputs, labels

        print('Epoch time: {}'.format(time.time() - _time_epoch))
        if epoch % save_every == 0:  # save the model every X epochs
            state_dic = {
                'epoch': epoch,
                'i_num_tot': i_num_tot,
                'i_num_epoch': i_num_epoch,
                'loss_output': loss_output,
                # 'loss_pre_ref': loss_pre_ref,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }
            torch.save(state_dic, model_dir + 'model.pth')

        log = '[epoch: {:d}/{:d}, ite: {:d}] loss_output: {:.6f}, l: {:.6f}\n'.format(
                epoch, epoch_num, i_num_tot, 
                loss_output / i_num_epoch, 
                loss[0].item()
            )

        del loss

        loss_output = 0
        loss_pre_ref = 0
        i_num_epoch = 0
        log_file = open('logs/log.txt', 'a+')
        log_file.write(log + '\n')
        log_file.close()
        print(log)

        if validation:
            model.eval()
            # val_i_num_tot = 0
            val_i_num_epoch = 0
            val_loss_output = 0
            # val_loss_pre_ref = 0
            val_log_file = open('logs/log_val.txt', 'a+')
            print('Evaluating...')
            with torch.no_grad():
                for val_i, val_data in enumerate(val_dataloader):
                    # val_i_num_tot += 1
                    val_i_num_epoch += 1

                    val_inputs, val_labels = val_data

                    val_inputs = val_inputs.cuda()
                    val_labels = val_labels.cuda()

                    val_out = model(val_inputs)

                    val_loss = muti_bce_loss_fusion(val_out, val_labels)

                    val_loss_output += val_loss[0].item()
                    # val_loss_pre_ref += val_loss0.item()

                    pred = val_out[0][:,0,:,:]
                    pred = normPRED(pred)

                    save_output(val_img_name_list[val_i], pred, prediction_dir)

                    del val_out, val_inputs, val_labels, val_loss

            log_val = '[val: epoch: {:d}, ite: {:d}] loss_output: {:.6f}\n'.format(
                    epoch, i_num_tot, val_loss_output / val_i_num_epoch )
            val_log_file.write(log_val + '\n')
            val_log_file.close()


    _t = 'Training time: '+ str(time.time() - _s) + '\n'
    print(_t)
    log_file = open('logs/log.txt', 'a+')
    log_file.write(_t)
    log_file.close()


if __name__ == '__main__':
    main()