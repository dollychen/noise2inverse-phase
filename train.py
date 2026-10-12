print("importing modules")
import time
start = time.time()
print("importing sys")
import sys
sys.path.append("../noise2inverse/")
print("importing numpy")
import numpy as np
from pathlib import Path
#from noise2inverse import tiffs, noise, fig

print("importing tifffile")
import tifffile
from tqdm import tqdm

print("importing wandb")
import wandb

print("importing torch") #moving import here so that incorrect path is discovered before loading the dataloader
import torch
from torch.utils.data import DataLoader
import torch.nn as nn  

print("importing other modules that should be fast")
import datetime
import os
import csv
import shutil
import argparse
import yaml
from scripts.utils import percetile_stretch, psnr_skimage, ssim_skimage, network_setup

end = time.time()
print(f"importing modules took {end-start} seconds")

print("imported all modules")
# # Scale pixel intensities during training such that its values roughly occupy the range [0,1].
# # This improves convergence.
# data_scaling = 1/65536

# get current  timestamp
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


def train(params):
    # set parameters from params dict
    train_data_path = params["data_path"]
    train_data_list = params["train_data_list"]
    val_data_list = params.get("val_data_list", train_data_list)
    low_res_dir = params.get("low_res_dir", "low_res")
    high_res_dir = params.get("high_res_dir", "high_res")

    model_dir = params["model_path"]
    resume_training = params["resume_training"]
    num_splits = params["num_splits"]
    strategy = params["strategy"]
    batch_size = params["batch_size"]
    data_scaling = params["data_scaling"]
    epochs = params["epochs"]
    network_name = params["network"]
    multi_gpu = params["multi_gpu"]
    n_features = params["n_features"]
    input_channels = params["input_channels"]
    pred_dir = Path(str(model_dir).replace("models", "predictions"))

    #data
    crop_size = params["crop_size"]
    centre_weight = params["centre_weight"]

    if not multi_gpu: 
        gpu_id = params["gpu_id"]
        print(f"the gpu id is {gpu_id}")
        torch.cuda.set_device(gpu_id)

    #print all parameters in params 
    print("--------------------------------------------------------------parameters--------------------------------------------------------------:")
    for key, value in params.items():
        print(f"{key}: {value}")


    print("--------------------------------------------------------------checking paths--------------------------------------------------------------")
    # check all paths exists
    assert os.path.exists(train_data_path), f"train data path {train_data_path} does not exist"
    assert os.path.exists(train_data_list), f"train data list {train_data_list} does not exist"
    # check parent dir of model output path exists
    assert os.path.exists(os.path.dirname(model_dir)), f"output path {os.path.dirname(model_dir)} does not exist"
    # check parent dir of pred output path exists
    assert os.path.exists(os.path.dirname(pred_dir)), f"output path {os.path.dirname(pred_dir)} does not exist"
    print("--------------------------------------------------------------paths checked--------------------------------------------------------------")


    print("--------------------------------------------------------------loading train data--------------------------------------------------------------")
    print("importing noise2inverse dataset modules") #moving import here so that incorrect path is discovered before loading the dataset
    from noise2inverse.datasets import (
        TiffDataset,
        Noise2InverseDataset,
    )

    #load dataset using the list given
    train_datasets = [TiffDataset(f"{train_data_path}", train_data_list, channel=input_channels,
                                  low_res_dir=low_res_dir, high_res_dir=high_res_dir)]
    val_datasets = [TiffDataset(f"{train_data_path}", val_data_list, channel=input_channels,
                                low_res_dir=low_res_dir, high_res_dir=high_res_dir)]
    train_ds = Noise2InverseDataset(*train_datasets, strategy=strategy, crop_size=crop_size, center_weight=centre_weight)
    val_ds = Noise2InverseDataset(*val_datasets, strategy=strategy)


    print("--------------------------------------------------------------loading dataloader--------------------------------------------------------------")
    # Dataloader and network:
    train_dl = DataLoader(train_ds, batch_size, shuffle=True,)

    #load the training data set unshuffled so that the same image could be identified and used for verifications
    val_dl = DataLoader(val_ds, batch_size, shuffle=False)
    #get a list of file names from the training data list
    with open(val_data_list, 'r') as f:
        val_img_names = f.read().splitlines()


    print(f'number of slices in training is: {train_ds.num_slices}, \
          number of splits used is: {train_ds.num_splits}')
    
    print("--------------------------------------------------------------loading network--------------------------------------------------------------")
    #load network

    network, optim = network_setup(network_name, multi_gpu, n_features, n_input_channels=input_channels)

    #load pretrained model
    if resume_training:
        print("resuming training")
        model_path = params["model_path"]
        checkpoint = torch.load(model_path)
        network.load_state_dict(checkpoint['state_dict'])
        optim.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']

    
    # The dataset contains multiple input-target pairs for each slice. 
    # Therefore, we divide by the number of splits to obtain the effective number of epochs.
    train_epochs = epochs
    torch.cuda.empty_cache() # clear GPU memory


    create_dir = True
    print("starting training loop, total number of epochs: ", train_epochs)
    # training loop
    for epoch in range(train_epochs):
        # Training in this epoch starts here
        loss_cum = 0
        ssim_measure = 0
        psnr_measure = 0

        for (inp, tgt) in tqdm(train_dl):
            inp = inp.cuda(non_blocking=True) / data_scaling
            tgt = tgt.cuda(non_blocking=True) / data_scaling

            # this is 2.5 D TODO: check if this is correct
            if input_channels != 1:
                tgt = tgt[:,0:1,:,:]

            # Do training step with masking
            output = network(inp)
            if network_name == "dncnn":
                loss = nn.functional.mse_loss(inp-output, tgt)
            elif network_name == "unet":
                loss = nn.functional.mse_loss(output, tgt)
            loss_cum += loss.item() # sum up batch loss
            optim.zero_grad()
            loss.backward()
            optim.step()

            # calculate ssim, psnr
            for i in range(len(inp)):
                ssim_measure += ssim_skimage(tgt[i], output[i])
                psnr_measure += psnr_skimage(tgt[i], output[i])


        epoch_loss = loss_cum / len(train_dl) # average loss over all batches
        avg_ssim = ssim_measure / len(train_dl)
        avg_psnr = psnr_measure / len(train_dl)

        # once an epoch is done, create output directories
        # only create the results folder when the run is successful to save unneccessary folders
        if create_dir:
            model_dir.mkdir(parents = True, exist_ok=True) # create model directory
            pred_dir.mkdir(parents = True, exist_ok=True) # create pred output directory
            shutil.copy2(params["config_file"], model_dir / "train_config.yaml")
            shutil.copy2(train_data_list, model_dir / "train_files.txt")
            shutil.copy2(val_data_list, model_dir / "validation_files.txt")
            create_dir = False

        #Save network
        torch.save(
            {"epoch": int(epoch), "state_dict": network.state_dict(), "optimizer": optim.state_dict(), "epoch_loss": epoch_loss}, 
            model_dir / f"weights_epoch_{epoch}_{epoch_loss:.6f}.torch"
        )
        print(f"model saved to {model_dir / f'weights_epoch_{epoch}_{epoch_loss:.6f}.torch'}")

        #create a folder for each epoch output
        epoch_dir = pred_dir / f"epoch_{epoch}"
        epoch_dir.mkdir(parents = True, exist_ok=True)
        train_output_dir = epoch_dir / "train_output"
        train_output_dir.mkdir(parents = True, exist_ok=True)

        print('epoch finished, running inference to save a few images')

        # run validation
        with torch.no_grad():           
            val_loss_cum = 0
            val_image_count = 0
            val_ssim_cum = 0
            val_psnr_cum = 0
            # run prediction over the complete held-out validation set
            for i, (inp, tgt) in tqdm(enumerate(val_dl), total=len(val_dl)):
                torch.cuda.empty_cache() # clear GPU memory
                test_inp = inp.cuda(non_blocking=True) / data_scaling # scale the input
                test_tgt = tgt.cuda(non_blocking=True) / data_scaling
                test_output = network(test_inp)
                val_loss_cum += nn.functional.mse_loss(test_output, test_tgt, reduction="sum").item()
                for j, image in enumerate(test_output):
                    image_index = i * batch_size + j
                    test_out_np = image.detach().cpu().numpy().squeeze()
                    tifffile.imwrite(str(train_output_dir / val_img_names[image_index]), test_out_np)
                    val_ssim_cum += ssim_skimage(test_tgt[j], image)
                    val_psnr_cum += psnr_skimage(test_tgt[j], image)
                    val_image_count += 1

        val_loss = val_loss_cum / (val_image_count * test_output.shape[-2] * test_output.shape[-1])
        val_ssim = val_ssim_cum / val_image_count
        val_psnr = val_psnr_cum / val_image_count
        log_path = model_dir / "training_log.csv"
        with open(log_path, "a", newline="") as log_file:
            writer = csv.writer(log_file)
            if log_file.tell() == 0:
                writer.writerow(["epoch", "train_loss", "train_ssim", "train_psnr", "val_loss", "val_ssim", "val_psnr"])
            writer.writerow([epoch, epoch_loss, avg_ssim, avg_psnr, val_loss, val_ssim, val_psnr])

        wandb.log({"loss": epoch_loss,
                   "epoch": epoch,
                    "ssim": avg_ssim,
                    "psnr": avg_psnr,
                    "val_loss": val_loss,
                    "val_ssim": val_ssim,
                    "val_psnr": val_psnr})

    


        print("--------------------------------------------------------------training loop finished--------------------------------------------------------------")


if __name__ == "__main__":

    # set config file
    print("Starting training script")
    parser = argparse.ArgumentParser(description='Noise2Inverse training script')
    parser.add_argument('--config', type=str, default='./configs/test/train_config_kidney.yaml', help='path to config file')
    args = parser.parse_args()

    # load config file
    config = yaml.safe_load(open(args.config))

    # load wandb config
    job_name_prefix = config['job_name_prefix']
    job_name = config.get("run_name", f'{job_name_prefix}_{timestamp}') # job name to identify the run
    if config['test_run']:
        config['wandb_project'] = 'test_project' # set the wandb project to test_project for testing
    
    # convert data paths to Path objects
    config['data_path'] = Path(config['data_path'])
    config['train_data_list'] = Path(config['train_data_list'])
    if config.get('val_data_list'):
        config['val_data_list'] = Path(config['val_data_list'])
    config['config_file'] = str(Path(args.config).resolve())
    
    # load model information
    config['model_path'] = Path(config['model_path'])/job_name


    print("--------------------------------------------------------------initialising wandb--------------------------------------------------------------")
    # initialise wandb
    wandb.init(
        project=config['wandb_project'], # set the wandb project where this run will be logged
        # track hyperparameters and run metadata
        config=config,
        name=job_name,
        mode= config['wandb_run_mode'] # disable wandb for testing
        ) #set disabled to 'disabled' to disable wandb

    print(f'cuda is available: {torch.cuda.is_available()}')
    print("--------------------------------------------------------------starting training--------------------------------------------------------------")
    train(config)
