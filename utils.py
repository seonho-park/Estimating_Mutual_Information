import torch
import os
import glob
import shutil
import json
import argparse
from datetime import datetime
from pprint import pprint


def get_device():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("device:", device)
    return device 


def set_seed(seed, device):
    torch.manual_seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def config_backup_get_log(args, filename):
    if not os.path.isdir('./results'):
        os.mkdir('./results')
    if not os.path.isdir('./chpt'):
        os.mkdir('./chpt')

    # set result dir
    current_time = str(datetime.now())
    dir_name = '%s_%s'%(current_time, args.suffix)
    result_dir = 'results/%s'%dir_name

    if not os.path.isdir(result_dir):
        os.mkdir(result_dir)
        os.mkdir(result_dir+'/codes')

    # deploy codes
    files = glob.iglob('*.py')
    model_files = glob.iglob('./models/*.py')

    for file in files:
        shutil.copy2(file, result_dir+'/codes')
    for model_file in model_files:
        shutil.copy2(model_file, result_dir+'/codes/models')


    # printout information
    print("Export directory:", result_dir)
    print("Arguments:")
    pprint(vars(args))

    logger = open(result_dir+'/%s.txt'%dir_name,'w')
    logger.write("%s \n"%(filename))
    logger.write("Export directory: %s\n"%result_dir)
    logger.write("Arguments:\n")
    logger.write(json.dumps(vars(args)))
    logger.write("\n")

    return logger, result_dir, dir_name


def process_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1000, help='random seed')
    parser.add_argument('--dim', type=int, default=128, help='number of dimensions [%(default)d]')
    parser.add_argument('--rho', type=float, default=0.9, help='correlation coefficient [%(default)g]')
    parser.add_argument('--suffix', type=str, default='test', help='suffix of result directory')
    args = parser.parse_args()
    
    return args
