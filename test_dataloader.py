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

Rain_Rain200H_test_path = '/data/ProjectData/Derain/Rain200H/test/'
Rain_Rain200H_train_path = '/data/ProjectData/Derain/Rain200H/train/'

Rain_SPA_test_path = '/data/ProjectData/Derain/DID-Data/Test/'
Rain_SPA_train_path = '/data/ProjectData/Derain/DID-Data/Train/'


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
    def __init__(self, data_style='train'):
        data_path = eval(f'Rain_Rain200H_{data_style}_path')
        super(Dataset_Rain200, self).__init__()
        self.data_path = data_path
        self.target_names =  data_path+'/norain/'
        self.input_names =  data_path+'/rain/'
        self.ls1 = os.listdir(data_path + "rain")
    def __len__(self):
        return len(self.ls1)
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

class Dataset_Rain200H(Dataset_Rain200):
    pass 


class Dataset_Rain100(udata.Dataset):
    def __init__(self, data_style='train'):
        data_path = eval(f'Rain_Rain200H_{data_style}_path')
        super(Dataset_Rain100, self).__init__()
        self.data_path = data_path
        self.target_names =  data_path+'/norain/'
        self.input_names =  data_path+'/rain/'
        self.ls1 = os.listdir(data_path + "rain")
    def __len__(self):
        return len(self.ls1)
    def __getitem__(self, index):
        target_names = self.target_names + 'norain-%03d.png'%(index+1)
        input_names = self.input_names + 'rain-%03d.png'%(index+1)
            
        y_origin = cv2.imread(os.path.join(input_names))
        gt = cv2.imread(os.path.join(target_names))
        gt = progress(gt)
        y = progress(y_origin)
        return y, gt




class Dataset_DID(udata.Dataset):
    def __init__(self, data_style):
        super(Dataset_DID, self).__init__()
        data_path = eval(f'Rain_DID_{data_style}_path')
        target_names =  data_path + 'norain/' 
        input_names =  data_path+'rain/'
        ls1 = os.listdir(data_path + "rain")
        # ls1 = [target_names+'/'+i for i in ls1]
        self.targets =  [target_names + i for i in ls1]
        self.inputs =  [input_names + i for i in ls1]
    def __len__(self):
            return len(self.targets)
    def __getitem__(self, index):
        target_names = self.targets[index]
        input_names = self.inputs[index]
        y_origin = cv2.imread(input_names)
        # print(input_names,target_names)
        gt = cv2.imread(os.path.join(target_names))
        gt = progress(gt).squeeze()
        y = progress(y_origin).squeeze()
        # print(y.shape, gt.shape)

        return y, gt
