import cv2
import os
import argparse
import numpy as np
import torch
from networks import CTPnet
import time
import tqdm
import torchvision.utils as utils
import codeHHF.test_dataloader as test_dataloader
from utils import SSIM,normalize,batch_PSNR
from torch.utils.data import DataLoader
import yaml
from yaml import Loader

progress = test_dataloader.progress

# set yaml path and load yaml config
cfg_path = "/home/jack/Project/VoiceIdentify1/ECAPA-HHF/YAML/test_pd.yaml"
cfg = yaml.load(open(cfg_path, "r").read(), Loader=Loader)
cfg_name = cfg_path.split('/')[-1].replace(".yaml","")
print("\n"*10+"Now running "+ cfg_name +"\n"*5)

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
parser = argparse.ArgumentParser(description="PReNet_Test1")
parser.add_argument("--use_GPU", type=bool, default=False, help='use GPU or not')
parser.add_argument("--recurrent_iter", type=int, default=4, help='number of recursive stages')
parser.add_argument("-Log_path", type=str, default='/data/ProjectData/Derain/Rain200L/TrainedModel/mixDTPNet/Logs', help='number of recursive stages')
opt = parser.parse_args()
# add yaml config into opt
for name in cfg:
  vars(opt)[name] = cfg[name]
  opt.result_list = opt.result_list.format(cfg_path.split('/')[-1].replace(".yaml",""))
os.system()

def test(model,model_path,data_name,test_all,name_index):
    '''
        model: 
            Net for tested.
        model_path: 
            The dir of paras.
        datapath: 
            Dir of the data.
        test_all: bool 
            Test all saved models or latest models paras.
        name_index: int 
            The model index.
    '''

    data_path = f'/data/ProjectData/Derain/{data_name}/Test/'
    data_set = eval(f"test_dataloder.Dataset_{data_name}({data_path})")
    Log_path = test_dataloader.Log_path + f'{data_name}'

    test_loader = DataLoader(dataset=data_set, batch_size=1, shuffle=False, num_workers=0, drop_last=False  )
    model = model.cuda()
    # 判断加载 单个模型/最后一个模型
    if test_all == True:
        load_name = os.path.join(model_path, f'net_epoch{name_index}.pth')
    elif model_path is not None:
        load_name = os.path.join(model_path, 'net_latest.pth')
    model.load_state_dict(torch.load(load_name))
    print('load_model from ' + load_name)
    model.eval()
    psnr_test ,pixel_metric,count,psnr_max,ssim_max = 0,0,0,0,0
    with torch.no_grad(): 
        if opt.use_GPU:
            torch.cuda.synchronize()
        times = 0
        for rainy, gt in test_loader:
            begin = time.time()
            out, _ = model(torch.squeeze(rainy,0))
            endtime = time.time()
            out = torch.clamp(out, 0., 1.)
            criterion = SSIM()
            loss = criterion(out, gt) * out.shape[0]
            pixel_metric += loss
            psnr_cur = batch_PSNR(out,  gt, 1.) * out.shape[0]
            psnr_test += psnr_cur
            if psnr_cur >= psnr_max:
                psnr_max = psnr_cur
            if loss >= ssim_max:
                ssim_max = loss
            count += out.shape[0]

            Note=open('/data/ProjectData/Derain/zlyexperiment/view/RID/out/test.txt','a')
            Note.write("[Test SSIM is] %0.2f, [Test PSNR is] %0.2f  [Current PSNR]%0.2f [max ]%0.2f maxssim %0.2f==================" % (pixel_metric/ count, psnr_test/ count,psnr_cur,psnr_max,ssim_max))
            Note.write('\n')
            print("[Test SSIM is] %0.2f, [Test PSNR is] %0.2f  [Current PSNR]%0.2f [max ]%0.2f maxssim %0.2f==================" % (pixel_metric/ count, psnr_test/ count,psnr_cur,psnr_max,ssim_max))

            # 选择是否把图像记录下来 图像记录在Log/result 结果记录在Log/{YAML}-result.txt
            if 0: # 输出图像
                x = utils.make_grid(torch.cat((out,gt,torch.squeeze(out_o,0))))
                x = np.uint8(255 * x.cpu().numpy().squeeze())
                r, g, b = cv2.split(x.transpose(1, 2, 0))
                cv2.imwrite(f'/data/ProjectData/Derain/zlyexperiment/view/RID/out/{file}',cv2.merge([b ,g, r]))
                x = utils.make_grid(out)
                x = np.uint8(255 * x.cpu().numpy().squeeze())
                r, g, b = cv2.split(x.transpose(1, 2, 0))
                cv2.imwrite(f'/data/ProjectData/Derain/zlyexperiment/view/RID/{file}',cv2.merge([b ,g, r]))
            times += (endtime - begin)
    psnst_average = psnr_test / count
    pixel_metric_average = pixel_metric / count
    return psnst_average.item(),psnr_max.item(),pixel_metric_average.item(),ssim_max.item(),times/count 
    # return times/count 

if __name__ == "__main__":

    model = CTPnet(recurrent_iter=3, use_GPU=True).cuda()
    res = []
    count = 0
    modelpath = "/data/ProjectData/Derain/Rain200L/TrainedModel/mixDTPNet/Logs/200L-SSIMtrick"
    datapath = "/data/ProjectData/Derain/Rain100H/rainy"
    res += test(model,modelpath,"DID",True,count)
    print(res)    
    