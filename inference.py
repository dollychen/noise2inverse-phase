from pathlib import Path
from noise2inverse import tiffs, noise, fig
from noise2inverse.datasets import (
    TiffDataset,
    Noise2InverseDataset,
)

from torch.utils.data import (
    DataLoader,
    Dataset
)
import torch
from scripts.utils import *
from tqdm import tqdm
import os
import tifffile
import yaml
import shutil
import datetime
import time

import argparse

timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

def predict(config_file):

    # load config file
    params = yaml.safe_load(open(config_file))

    inf_data_path = Path(params["data_path"])
    inf_data_list = Path(params["data_list"])

    #pred_output_path = Path(params["pred_output_path"])
    model_path = Path(params["model_path"])
    checkpoint_name = params["checkpoint_name"]
    #dataset_name = params["dataset_name"]
    #batch_size = params["num_splits"]
    num_splits = params["num_splits"]
    strategy = params["strategy"]
    network_name = params["network"]
    data_scaling = params["data_scaling"]
    n_features = params["n_features"]
    input_channels = params["input_channels"]

    multi_gpu = params["multi_gpu"]

    pred_dir = Path(str(model_path).replace("models", "predictions"))

    save_split_pred = params['save_split_pred']
    
   

    if not multi_gpu:
        gpu_id = params["gpu_id"]
        print(f"the gpu id is {gpu_id}")
        torch.cuda.set_device(gpu_id)

        #print all parameters in params 
    print("--------------------------------------------------parameters:--------------------------------------------------")
    for key, value in params.items():
        print(f"{key}: {value}")

    #check paths exists
    assert os.path.exists(inf_data_path), f"test data path {inf_data_path} does not exist"
    assert os.path.exists(inf_data_list), f"test data list {inf_data_list} does not exist"
    assert os.path.exists(model_path), f"model path {model_path} does not exist"
    assert os.path.exists(model_path / checkpoint_name), f"model {checkpoint_name} does not exist"

    print("--------------------------------------------------paths checked--------------------------------------------------")
    model_name = os.path.basename(model_path)
    print(f"model name is: {model_name}")



    print("--------------------------------------------------loading data--------------------------------------------------")
    test_datasets = [TiffDataset(f"{inf_data_path}/{j}", inf_data_list, channel=input_channels) for j in range(num_splits)]
    test_ds = Noise2InverseDataset(*test_datasets, strategy=strategy)

    # Dataloader and network:
    test_dl = DataLoader(test_ds, num_splits, shuffle=False) #note batchsize is the same as num_split 
    with open(inf_data_list, 'r') as f:
        test_img_names = f.read().splitlines()

    print("setting network")
    network, _ = network_setup(network_name, multi_gpu, n_features)


    # load model
    print('loading weights:', model_path / checkpoint_name)
    checkpoint = torch.load(model_path / checkpoint_name)
    if multi_gpu:
        
        #network.load_state_dict(checkpoint['state_dict'])
        network.module.load_state_dict(checkpoint['state_dict'])
        
    else:

        #print loading multi gpu model to single gpu
        state_dict = checkpoint['state_dict']
        new_state_dict = {}
        for key in state_dict:
            new_key = key.replace('module.','')
            new_state_dict[new_key] = state_dict[key]
        network.load_state_dict(new_state_dict)

    epoch = checkpoint['epoch']
    print(f"loaded model from epoch {epoch}")

    create_dir = True # only create output directory when prediction is ready

    # predict
    with torch.no_grad():
        network.eval() # set model to evaluation mode
        print("start prediction")

        # run the prediction
        for i, (inp, _) in tqdm(enumerate(test_dl)):
            start = time.time()
            torch.cuda.empty_cache() # clear GPU memory
            test_inp = inp.cuda(non_blocking=True) / data_scaling # scale the input
            test_output = network(test_inp)

            if create_dir:
                #create output directory if not exist
                

                epoch_dir = pred_dir / f"epoch_{epoch}"
                epoch_dir.mkdir(parents = True, exist_ok=True)
                inf_output_dir = epoch_dir / f"inf_output_{timestamp}"
                inf_output_dir.mkdir(parents = True, exist_ok=True)
                avg_output_dir = inf_output_dir / "avg"
                avg_output_dir.mkdir(parents = True, exist_ok=True)
                shutil.copy(config_file, f"{avg_output_dir}/pred_config.yaml")
                if save_split_pred:
                    #make sub dire for split
                    split_0_dir = inf_output_dir / "0"
                    split_0_dir.mkdir(parents=True, exist_ok=True)
                    split_1_dir = inf_output_dir / "1"
                    split_1_dir.mkdir(parents=True, exist_ok=True)
                create_dir = False


            if save_split_pred:
                inf_out_0_np = (test_output[0] * data_scaling).detach().cpu().numpy().squeeze()
                inf_out_1_np = (test_output[1] * data_scaling).detach().cpu().numpy().squeeze()
                #save output image
                out_path_0 = str(split_0_dir / f"{test_img_names[i]}")
                out_path_1 = str(split_1_dir / f"{test_img_names[i]}")
                tifffile.imsave(out_path_0, inf_out_0_np)
                tifffile.imsave(out_path_1, inf_out_1_np)

            test_output_img = test_output.mean(axis = 0) * data_scaling # averaged over batch size as it is the number of splits
            
            #save output image
            test_out_np = (test_output_img).detach().cpu().numpy().squeeze()
            out_path = str(avg_output_dir / f"{test_img_names[i]}")
            tifffile.imsave(out_path, test_out_np)
            end = time.time()
            #print(f"prediction time for {test_img_names[i]} is {end - start}")

    print("--------------------------------------------------------------prediction finished--------------------------------------------------------------")
if __name__ == "__main__":

    print("parse arguments")

    parser = argparse.ArgumentParser(description='Noise2Inverse inference script')
    parser.add_argument('--config', type=str, default='./configs/kidney/inf_config_kidney.yaml', help='path to config file')
    args = parser.parse_args()


    # set parameters
    
    print("start inference")
    predict(args.config)