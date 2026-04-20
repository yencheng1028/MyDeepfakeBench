"""
eval pretained model.
"""
import os
import numpy as np
from os.path import join
import cv2
import random
import datetime
import time
import yaml
import pickle
from tqdm import tqdm
from copy import deepcopy
from PIL import Image as pil_image
from metrics.utils import get_test_metrics
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.utils.data
import torch.optim as optim

from dataset.abstract_dataset import DeepfakeAbstractBaseDataset
from dataset.ff_blend import FFBlendDataset
from dataset.fwa_blend import FWABlendDataset
from dataset.pair_dataset import pairDataset

from trainer.trainer import Trainer
from detectors import DETECTOR
from metrics.base_metrics_class import Recorder
from collections import defaultdict

import argparse
from logger import create_logger

# 建立一個 Command-line argument parser物件，呼叫 argparse 套件中的constructor(類別建構子)建立
parser = argparse.ArgumentParser(description='Process some paths.')
# 定義可接受的參數
parser.add_argument('--detector_path', type=str, 
                    default='/home/ernest/MyDeepfakeBench/training/config/detector/resnet34.yaml',
                    help='path to detector YAML file')
parser.add_argument("--test_dataset", nargs="+")                    # nargs: 可輸入多個值                                  
parser.add_argument('--weights_path', type=str, 
                    default='/home/ernest/MyDeepfakeBench/training/weights/xception_best.pth')
# parser.add_argument("--lmdb", action='store_true', default=False)
args = parser.parse_args()                                          # 讀取在終端機輸入的所有內容

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def init_seed(config):
    # 檢查設定檔中是否指定種子編號
    if config['manualSeed'] is None:
        config['manualSeed'] = random.randint(1, 10000)
    random.seed(config['manualSeed'])                              # 設定 Python 隨機模組的種子
    torch.manual_seed(config['manualSeed'])                         # 設定 PyTorch (CPU 運算) 的隨機編號種子
    if config['cuda']:
        torch.cuda.manual_seed_all(config['manualSeed'])         # 設定所有 GPU 的隨機編號種子，確保顯示卡上的運算也是可重現的

# 將要測試的資料集打包成 PyTorch 可以讀取的格式
def prepare_testing_data(config):
    
    # 負責產生單一一個資料集的載入器
    def get_test_data_loader(config, test_name):
        config = config.copy()                             # 複製一份出來修改                
        config['test_dataset'] = test_name                 # 設定當前要處理的資料集         
        
        # Class Instance，建立一個 Dataset 物件，負責去硬碟找出圖片路徑、標籤
        test_set = DeepfakeAbstractBaseDataset(
                config=config,
                mode='test', 
            )

        # 建立 PyTorch 的 DataLoader，負責批次 (batch) 傳送圖片給模型    
        test_data_loader = \
            torch.utils.data.DataLoader(
                dataset=test_set, 
                batch_size=config['test_batchSize'],
                shuffle=False, 
                num_workers=int(config['workers']),
                collate_fn=test_set.collate_fn,
                drop_last=False                                 # 如果最後剩下的圖片不夠一個批次，也不要捨棄
            )
        return test_data_loader

    # 遍歷所有資料集
    test_data_loaders = {}                                  # 空字典用來存放多個載入器
    for one_test_name in config['test_dataset']:
        test_data_loaders[one_test_name] = get_test_data_loader(config, one_test_name)
    return test_data_loaders


def choose_metric(config):
    metric_scoring = config['metric_scoring']
    if metric_scoring not in ['eer', 'auc', 'acc', 'ap']:
        raise NotImplementedError('metric {} is not implemented'.format(metric_scoring)) # 強制停止程式並顯示錯誤訊息
    return metric_scoring


def test_one_dataset(model, data_loader):
    prediction_lists = []
    feature_lists = []
    label_lists = []

    # tqdm (進度條) len(data_loader): 總共有多少批次
    for i, data_dict in tqdm(enumerate(data_loader), total=len(data_loader)):
        # get data
        data, label, mask, landmark = \
        data_dict['image'], data_dict['label'], data_dict['mask'], data_dict['landmark']
        
        # 語法：torch.where(條件, 條件成立則改為此值, 不成立則改為此值)
        # 意義：這是一行「標籤標準化」。不論原始標籤是多少，只要不等於 0 ，一律設為 1 
        label = torch.where(data_dict['label'] !=0, 1, 0)
        
        # move data to GPU
        data_dict['image'], data_dict['label'] = data.to(device), label.to(device)
        if mask is not None:
            data_dict['mask'] = mask.to(device)
        if landmark is not None:
            data_dict['landmark'] = landmark.to(device)

        # .cpu(): 將資料從 GPU 搬回 CPU
        # .detach(): 將資料從計算圖中分離（因為測試不需要計算梯度）
        # .numpy(): 轉換成 numpy 格式，方便後續計算指標
        # list() 與 += : 將這一批次的結果併入總清單中
        predictions = inference(model, data_dict)
        label_lists += list(data_dict['label'].cpu().detach().numpy())
        prediction_lists += list(predictions['prob'].cpu().detach().numpy())
        feature_lists += list(predictions['feat'].cpu().detach().numpy())
    
    return np.array(prediction_lists), np.array(label_lists),np.array(feature_lists)
    
def test_epoch(model, test_data_loaders):
    # set model to eval mode                         這會關閉只在訓練時使用的功能（如 Dropout 或 Batch Normalization）
    model.eval()

    # 用來存放所有資料集測試後的結果分數
    metrics_all_datasets = {}

    # testing for all test data
    keys = test_data_loaders.keys()                    # 語法：dict.keys() 取得字典所有的鍵（也就是所有資料集的名稱）
    for key in keys:
        # 語法：物件屬性存取
        # 意義：從 DataLoader 中挖出原始的資料字典，為了後面拿圖片檔名（img_names）
        data_dict = test_data_loaders[key].dataset.data_dict
        # compute loss for each dataset
        predictions_nps, label_nps, feat_nps = test_one_dataset(model, test_data_loaders[key])
        
        # compute metric for each dataset
        metric_one_dataset = get_test_metrics(y_pred=predictions_nps, y_true=label_nps,
                                              img_names=data_dict['image'])
        metrics_all_datasets[key] = metric_one_dataset                          # 儲存指標分數
        
        # info for each dataset
        tqdm.write(f"dataset: {key}")                            # 在不破壞進度條顯示的情況下，印出當前測試的資料集名稱
        for k, v in metric_one_dataset.items():
            tqdm.write(f"{k}: {v}")                             # 印出指標分數，例如 "auc: 0.98", "acc: 0.95"

    return metrics_all_datasets

# 語法：Decorator
# 意義：告訴 PyTorch 接下來這塊程式碼「不需要計算梯度 (Gradient)
# 因為測試時不需要更新模型參數，能大幅節省記憶體空間並加快運算速度
@torch.no_grad()
def inference(model, data_dict):

    # 語法：模型物件呼叫 (__call__)
    # 意義：執行模型的前向傳播 (Forward Pass)。
    # 傳入 inference=True 是一個自定義標記，通常讓模型知道現在是測試階段，只需回傳機率與特徵。
    predictions = model(data_dict, inference=True)
    
    # 回傳模型輸出的字典（包含 prob 預測機率、feat 特徵等）
    return predictions


def main():
    # parse options and load config
    with open(args.detector_path, 'r') as f:
        config = yaml.safe_load(f)                                     # 將 YAML 格式轉為 Python 的字典 (dict) 格式
    with open('./training/config/test_config.yaml', 'r') as f:
        config2 = yaml.safe_load(f)
    config.update(config2)                                              # 將 config2 的內容合併到 config 中
    
    if 'label_dict' in config:
        config2['label_dict']=config['label_dict']
    weights_path = None
    
    # 如果有輸入--test_dataset 參數，則覆蓋掉原本的設定
    if args.test_dataset:
        config['test_dataset'] = args.test_dataset
    if args.weights_path:
        config['weights_path'] = args.weights_path
        weights_path = args.weights_path
    
    # 固定隨機種子以確保實驗結果可重現
    init_seed(config)

    # set cudnn benchmark if needed
    if config['cudnn']:
        cudnn.benchmark = True                                      # 讓硬體針對卷積運算尋找最快演算法

    # prepare the testing data loader
    test_data_loaders = prepare_testing_data(config)
    
    # prepare the model (detector)
    # 從偵測器字典 (DETECTOR) 中，根據設定的 model_name 找出對應的模型
    model_class = DETECTOR[config['model_name']]       
    model = model_class(config).to(device)                      # 實例化模型並搬移到硬體 (GPU 或 CPU)
    epoch = 0
    if weights_path:
        # 嘗試從檔名解析這是第幾個訓練階段 (epoch) 的權重
        # 例如從 'ckpt_epoch_9_best.pth' 中切割字串取得數字 9
        try:
            epoch = int(weights_path.split('/')[-1].split('.')[0].split('_')[2])
        except:
            epoch = 0
        
        ckpt = torch.load(weights_path, map_location=device)            # 使用 PyTorch 載入權重檔案 (.pth)
        model.load_state_dict(ckpt, strict=True)             # 將權重注入模型。strict=True 代表模型結構與權重必須完全吻合
        print('===> Load checkpoint done!')
    else:
        print('Fail to load the pre-trained weights')
    
    # start testing
    best_metric = test_epoch(model, test_data_loaders)
    print('===> Test Done!')

# 當此檔案被直接執行 (而不是被當成模組 import) ，則觸發 main()
if __name__ == '__main__':
    main()
