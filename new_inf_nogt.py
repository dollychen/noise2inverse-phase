import torch
import tifffile
from skimage.metrics import structural_similarity as ssim_skimage, peak_signal_noise_ratio as psnr_skimage
from tqdm import tqdm
import yaml
from pathlib import Path
from noise2inverse.datasets import TiffDataset, Noise2InverseDataset
from torch.utils.data import DataLoader
import os
from scripts.utils import network_setup

# Extract parameters from config
data_scaling = 65535
data_path = "/cluster/project7/HiP_CT_Denoise/data/kidney_left/2021_17/25.14um/recon/orig_recon_joseph/all/full/25.14um_slice_matching_0_3088"
val_data_list = "./configs/test/some_kidney_data.txt"
model_path = "/cluster/project7/HiP_CT_Denoise/models/noise2inverse_phase/test-kidney_25um_pag_split2_unet32_highres_simple_20250304-015540"
inference_output_dir = Path("/cluster/project7/HiP_CT_Denoise/predictions/noise2inverse_phase/test-kidney_25um_pag_split2_unet32_highres_simple_20250304-015540")
inference_output_dir.mkdir(parents=True, exist_ok=True)
batch_size = 1
no_epochs = 109

#load dataset using the list given
val_datasets = [TiffDataset(f"{data_path}", val_data_list, channel=1, test=True)]
val_ds = Noise2InverseDataset(*val_datasets)


print("--------------------------------------------------------------loading dataloader--------------------------------------------------------------")


#load the training data set unshuffled so that the same image could be identified and used for verifications
val_dl = DataLoader(val_ds, batch_size, shuffle=False)

model_list = os.listdir(model_path)


# Inference loop
with torch.no_grad():
    for model_name in model_list:

        if "107" not in model_name:
            continue
        # Load the model
        network, _ = network_setup("unet", multi_gpu=False, n_features=64)

        checkpoint = torch.load(os.path.join(model_path,model_name), weights_only=True)

              #print loading multi gpu model to single gpu
        state_dict = checkpoint['state_dict']
        new_state_dict = {}
        for key in state_dict:
            new_key = key.replace('module.','')
            new_state_dict[new_key] = state_dict[key]
        network.load_state_dict(new_state_dict)

        epoch = checkpoint['epoch']

        network.eval()

        with open (val_data_list, 'r') as f:
            val_img_names = f.read().splitlines()
        epoch_dir = inference_output_dir / f"epoch_{epoch}"
        epoch_dir.mkdir(parents=True, exist_ok=True)
        epoch_output_dir = epoch_dir / "full_image_output"
        epoch_output_dir.mkdir(parents=True, exist_ok=True)

        print(f'Epoch {epoch}')

        image_ssim = []
        image_psnr = []

        for i, (low, _) in tqdm(enumerate(val_dl)):
            torch.cuda.empty_cache()  # clear GPU memory
            test_inp = low.cuda(non_blocking=True) / data_scaling  # scale the input
            test_output = network(test_inp)
            for j, image in enumerate(test_output):
                #save the output image
                test_out_np = (image * data_scaling).detach().cpu().numpy().squeeze()
                out_path = str(epoch_output_dir / f"{val_img_names[0]}")
                val_img_names.pop(0)
                tifffile.imwrite(out_path, test_out_np)

  
print("Inference completed.")