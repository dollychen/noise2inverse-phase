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
    train_datasets = [TiffDataset(f"{train_data_path}/{j}", train_data_list, channel=input_channels) for j in range(num_splits)]
    train_ds = Noise2InverseDataset(*train_datasets, strategy=strategy, crop_size=crop_size, center_weight=centre_weight)
    val_ds = Noise2InverseDataset(*train_datasets, strategy=strategy)


    print("--------------------------------------------------------------loading dataloader--------------------------------------------------------------")
    # Dataloader and network:
    train_dl = DataLoader(train_ds, batch_size, shuffle=True,)

    #load the training data set unshuffled so that the same image could be identified and used for verifications
    val_dl = DataLoader(val_ds, num_splits, shuffle=False)
    #get a list of file names from the training data list
    with open(train_data_list, 'r') as f:
        val_img_names = f.read().splitlines()
    # generate 10 images to check for over fitting
    np.random.seed(0)
    val_idx = np.random.randint(0,len(val_dl),size = 5)
    wandb_index = val_idx[np.random.choice(len(val_idx))]


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
    train_epochs = max(epochs // num_splits, 1)
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

            # run the prediction
            for i, (inp, tgt) in tqdm(enumerate(val_dl)):
                if i not in val_idx:
                    continue
                torch.cuda.empty_cache() # clear GPU memory
                test_inp = inp.cuda(non_blocking=True) / data_scaling # scale the input
                test_output = network(test_inp)
                test_output_img = test_output.mean(axis = 0) * data_scaling # averaged over batch size as it is the number of splits

                #save output image
                test_out_np = (test_output_img).detach().cpu().numpy().squeeze()
                out_path = str(train_output_dir / f"{val_img_names[i]}")
                tifffile.imwrite(out_path, test_out_np)   

                # calculate ssim
                image_ssim = ssim_skimage(tgt[0], test_output[0])
                image_psnr = psnr_skimage(tgt[0], test_output[0])

                #save the image to wandb
                if i == wandb_index:
                    test_img_name = val_img_names[i]
                    image_in = wandb.Image(percetile_stretch(inp[0] * data_scaling), caption=f"input {test_img_name}")
                    image_out = wandb.Image(percetile_stretch(test_output[0] * data_scaling), caption=f"output psnr: {image_psnr:.2f}, ssim: {image_ssim:.2f}")
                    image_tgt = wandb.Image(percetile_stretch(tgt[0] * data_scaling), caption=f"target")
                    image_avg = wandb.Image(percetile_stretch(test_output_img), caption="average output")


        wandb.log({"loss": epoch_loss, 
                   "images": (image_in, image_tgt, image_out, image_avg), 
                   "epoch": epoch,
                    "ssim": avg_ssim,
                    "psnr": avg_psnr})

    


        print("--------------------------------------------------------------training loop finished--------------------------------------------------------------")


if __name__ == "__main__":

    # set config file
    print("Starting training script")
    parser = argparse.ArgumentParser(description='Noise2Inverse training script')
    parser.add_argument('--config', type=str, default='./configs/kidney/train_config_kidney.yaml', help='path to config file')
    args = parser.parse_args()

    # load config file
    config = yaml.safe_load(open(args.config))

    # load wandb config
    job_name_prefix = config['job_name_prefix']
    job_name = f'{job_name_prefix}_{timestamp}' # job name to identify the run
    if config['test_run']:
        config['wandb_project'] = 'test_project' # set the wandb project to test_project for testing
    
    # convert data paths to Path objects
    config['data_path'] = Path(config['data_path'])
    config['train_data_list'] = Path(config['train_data_list'])
    
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