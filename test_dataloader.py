import torch.utils.data as udata
import cv2
import os
from utils import SSIM,normalize,batch_PSNR
import numpy as np
import torch
from patchify import patchify

Log_path = "/data/ProjectData/Derain/Rain200L/TrainedModel/mixDTPNet/Logs"
   
# 模型地址
I_HAZE_log_path = '/home/jack/Project/Derain/CTPNet/hjhDerain/Logs/Train_ITS_2'
O_HAZE_log_path = '/home/huangjiehui/Project/DerainNet/Train_OTS2'
SOTS_I_log_path = '/home/jack/Project/Derain/CTPNet/hjhDerain/Logs/Train_ITS_2'
DID_data_path = '/data1/hjh/62190446236f408cbb1b3bb08c8b1241/JackFiles/ProjectData/RainData/DeRain/DID-Data/DID-MDN-test'
data_path_800 = '/data1/hjh/62190446236f408cbb1b3bb08c8b1241/JackFiles/ProjectData/RainData/DeRain/Rain800/Rain800/rain800_test'

path_200L = '/home/jack/Project/Derain/CTPNet/hjhDerain/Logs/200L'
path_X2_H_patch  = '/home/jack/Project/Derain/CTPNet/hjhDerain/Logs/X2_H_patch'
X2path = "/home/jack/Project/Derain/CTPNet/hjhDerain/Logs/X2"
path_X2_H = '/home/jack/Project/Derain/CTPNet/hjhDerain/Logs/X2_H'
path_JN_14000 = '/home/jack/Project/Derain/CTPNet/hjhDerain/Logs/JN_14000'
path_20H = "/home/jack/Project/Derain/CTPNet/hjhDerain/Logs/X2_H"
path_800 = '/home/jack/Project/Derain/CTPNet/hjhDerain/Logs/_800'
path_did = '/home/jack/Project/Derain/CTPNet/hjhDerain/Logs/DID'
path_did_patch = '/home/jack/Project/Derain/CTPNet/hjhDerain/Logs/DID_patch_192_L'
path_14000_patch = '/home/jack/Project/Derain/CTPNet/hjhDerain/Logs/AM2_14000'
path_DID_patch_192_L = '/home/jack/Project/Derain/CTPNet/hjhDerain/Logs/DID_patch_192_L'
path_200H_patch = '/home/huangjiehui/Project/DerainNet/Logs/200H_patch'
path_200H = '/home/jack/Project/Derain/CTPNet/hjhDerain/Logs/200H'
path_X2_H_patch ='/home/huangjiehui/Project/DerainNet/Logs/X2_H_patch_2'
path_OTS_o = "/home/jack/Project/Derain/CTPNet/hjhDerain/Logs/OTS"
#  数据地址
Rain_200H_data_path = '/data1/hjh/62190446236f408cbb1b3bb08c8b1241/JackFiles/ProjectData/RainData/DeRain/Rain200H/Rain200H/test'
Rain_200L_data_path = '/data/ProjectData/Derain/Rain200L/test'
Rain_100H_data_path = '/data/ProjectData/Derain/Rain100H'
Rain_100L_data_path = '/data/ProjectData/Derain/Rain100L'

Rain_DID_test_path = '/data/ProjectData/Derain/DID-Data/Test/'
Rain_DID_train_path = '/data/ProjectData/Derain/DID-Data/Train/'

#  I-HAZE
I_HAZE_data_path = '/home/huangjiehui/Project/DerainNet/JackData/I-HAZE/hazy'

#  O-HAZE
O_HAZE_data_path = '/data1/hjh/ProjectData/Defogging/O-HAZE/hazy'

#  SOTS-Ondoor
SOTS_Ondoor_data_path = '/data1/hjh/ProjectData/Defogging/SOTS/Outdoor/hazy'
SOTS_O_log_path = '/home/jack/Project/Derain/CTPNet/hjhDerain/Logs/Train_OTS2'

#  SOTS-Indoor
SOTS_Indoor_data_path = '/data1/hjh/ProjectData/Defogging/SOTS/Indoor/hazy'



def progress(y_origin):
    '''将cv2读到的图片转化到torch类型，长宽符合条件的RGB类型
        :typt y_origin: List[List[int]]
            输入原始image图片 y_origin int类型 通道为BGR.
        :rtype: Tensor(List[List[int]])
            输出为长宽为32倍数的 torch 形式 通道为RGB
    '''
    #  rbg 255 add_channel 32
    b, g, r = cv2.split(y_origin)
    y = cv2.merge([r, g, b])
    
    y = normalize(np.float32(y))
    y = np.expand_dims(y.transpose(2, 0, 1), 0)
    
    high = y.shape[2]//32
    wight = y.shape[3]//32
    y = y[:,:, 0:high*32, 0:wight*32]
    y = torch.Tensor(y)
    return y


class Dataset_Rain200(udata.Dataset):
    def __init__(self, data_path='.'):
        super(Dataset_Rain200, self).__init__()
        self.data_path = data_path
        if ('200' in self.data_path):
            self.target_names =  data_path+'/norain/'
            self.input_names =  data_path+'/rain/'
        elif  ('100' in self.data_path):
            self.target_names =  data_path + '/'
            self.input_names =  data_path+'/rainy/'
    def __len__(self):
        return len(self.input_names)
    def __getitem__(self, index):
        if ('200' in self.data_path):
            target_names = self.target_names + f'norain-{index+1}.png'
            input_names =  self.input_names + f'norain-{index+1}x2.png'
        elif  ('100' in self.data_path):
            target_names = self.target_names + 'norain-%03d.png'%(index+1)
            input_names = self.input_names + 'rain-%03d.png'%(index+1)
            
        y_origin = cv2.imread(os.path.join(input_names))
        gt = cv2.imread(os.path.join(target_names))
        gt = progress(gt)
        y = progress(y_origin)
        return y, gt
#  ITS
# ITS_data_path = '/home/huangjiehui/Project/DerainNet/JackData/ITS/train'
# log_path = '/home/huangjiehui/Project/DerainNet/Logs'

class Dataset_Rain200H(Dataset_Rain200):
    pass 

class Dataset_Rain100(udata.Dataset):
    def __init__(self, data_path='.'):
        super(Dataset_Rain100, self).__init__()
        self.data_path = data_path
        if ('200' in self.data_path):
            self.target_names =  data_path+'/norain/'
            self.input_names =  data_path+'/rain/'
        elif  ('100' in self.data_path):
            self.target_names =  data_path + '/norain/'
            self.input_names =  data_path+'/rainy/'
    def __len__(self):
        return len(self.input_names)
    def __getitem__(self, index):
        if ('200' in self.data_path):
            target_names = self.target_names + f'norain-{index+1}.png'
            input_names =  self.input_names + f'norain-{index+1}x2.png'
        elif  ('100' in self.data_path):
            target_names = self.target_names + 'norain-%03d.png'%(index+1)
            input_names = self.input_names + 'rain-%03d.png'%(index+1)
            
        y_origin = cv2.imread(os.path.join(input_names))
        gt = cv2.imread(os.path.join(target_names))
        gt = progress(gt)
        y = progress(y_origin)
        return y, gt

class Dataset(udata.Dataset):
    def __init__(self, data_path=SOTS_Ondoor_data_path):
        super(Dataset, self).__init__()
        self.data_path = data_path
        self.img_names =  os.listdir(data_path)
        self.num = 6    # 把一张一张大图分为num份
    def __len__(self):
        if ('O-HAZE' in self.data_path or 'I-HAZE'in self.data_path) :
            return len(self.img_names)*self.num
        return len(self.img_names)
    def __getitem__(self, index):
        # target
        # GT path with target data
        if 'O-HAZE' in self.data_path or 'I-HAZE'in self.data_path :
            # !!!!!!RGB
            y_origin = cv2.imread(os.path.join(self.data_path, self.img_names[index//self.num]))
            gt_path = os.path.join(self.data_path, self.img_names[index//self.num]).replace("hazy",'GT')   # #  I-HAZE O-HAZE
            gt = cv2.imread(gt_path)
            gt= patchify(gt.transpose(2, 0, 1),(3,96,96),step=96).reshape(-1,3,96,96)/255
            y= patchify(y_origin.transpose(2, 0, 1),(3,96,96),step=96).reshape(-1,3,96,96)/255
            # gt,y = progress(gt),progress(y)
            l = gt.shape[0]//(self.num-1)
            return y[index%self.num*l : (index%self.num+1)*l ,:,:,:], gt[index%self.num*l : (index%self.num+1)*l ,:,:,:]
        
        elif ('SOTS/O' in self.data_path or 'SOTS/I'in self.data_path) :
            y_origin = cv2.imread(os.path.join(self.data_path, self.img_names[index]))
            gt_path = os.path.join(self.data_path.replace('hazy','nohazy'), self.img_names[index].split('_')[0]+'.png') #  SOTS-Ondoor SOTS-Indoor 
            # gt = progress(cv2.imread(gt_path))
            y = progress(y_origin)
            if 'SOTS/Indoor' in self.data_path:
                gt = progress(cv2.imread(gt_path))[:,:,10:458,10:618]  # #  SOTS-Indoor
            else:
                gt = progress(cv2.imread(gt_path))
            return y, gt

class Dataset_DID_800(udata.Dataset):
    def __init__(self, data_path='.'):
        super(Dataset_DID_800, self).__init__()
        self.data_path = data_path
        ls1 = os.listdir(data_path)
        ls1 = [data_path+'/'+i for i in ls1]
        self.keys = ls1
    def __len__(self):
        if 'DID' in self.data_path:
            return 1000
        else:
            return len(self.keys)
    def __getitem__(self, index):
        key = self.keys[index]
        img = cv2.imread(key)
        b, g, r = cv2.split(img)
        img = cv2.merge([r, g, b])
        img = np.float32(normalize(img)).transpose(2,0,1)
        target = img[:,:,0:img.shape[2]//2]
        input = img[:,:,img.shape[2]//2:]
        target,input =np.expand_dims(target, 0), np.expand_dims(input, 0)

        high = target.shape[2]//32
        wight = target.shape[3]//32
        target = target[:,:, 0:high*32, 0:wight*32]
        
        high = input.shape[2]//32
        wight = input.shape[3]//32
        input = input[:,:, 0:high*32, 0:wight*32]
        return torch.Tensor(input), torch.Tensor(target)
