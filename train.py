import os
import json

json_dict = dict()
train_dict_list = list()

images = sorted(os.listdir('10602767/BONBID2023_Train/1ADC_ss'))
z_maps = sorted(os.listdir('10602767/BONBID2023_Train/2Z_ADC'))
labels = sorted(os.listdir('10602767/BONBID2023_Train/3LABEL'))

for i in range(len(images)):
     temp_dict = dict()

     image_path = os.path.join('10602767/BONBID2023_Train/1ADC_ss', images[i])
     zmap_path = os.path.join('10602767/BONBID2023_Train/2Z_ADC', z_maps[i])
     label_path = os.path.join('10602767/BONBID2023_Train/3LABEL', labels[i])

     temp_dict['image'] = image_path
     temp_dict['zmap'] = zmap_path
     temp_dict['label'] = label_path

     train_dict_list.append(temp_dict)

json_dict['training'] = train_dict_list

with open('./train_dataset.json', 'w') as file:
     json.dump(json_dict, file)

print("train_dataset.json file created successfully!")

json_dict = dict()
val_dict_list = list()

images = sorted(os.listdir('10602767/BONBID2023_Val/1ADC_ss'))
z_maps = sorted(os.listdir('10602767/BONBID2023_Val/2Z_ADC'))
labels = sorted(os.listdir('10602767/BONBID2023_Val/3LABEL'))

for i in range(len(images)):
     temp_dict = dict()

     image_path = os.path.join('10602767/BONBID2023_Val/1ADC_ss', images[i])
     zmap_path = os.path.join('10602767/BONBID2023_Val/2Z_ADC', z_maps[i])
     label_path = os.path.join('10602767/BONBID2023_Val/3LABEL', labels[i])
 
     temp_dict['image'] = image_path
     temp_dict['zmap'] = zmap_path
     temp_dict['label'] = label_path

     val_dict_list.append(temp_dict)

json_dict['validation'] = val_dict_list

with open('./val_dataset.json', 'w') as file:
     json.dump(json_dict, file)

print("val_dataset.json file created successfully!")

import json
import torch
import numpy as np
import time
from monai.transforms import (
    Compose, LoadImaged, NormalizeIntensityd, Activations,
    AsDiscrete, RandZoomd, RandRotated, RandGaussianNoised,
    ToTensord, RandCropd, RandSpatialCrop, RandFlipd, Rand3DElasticd
)
from monai.metrics import DiceMetric
from monai.networks.nets import SwinUNETR
from monai.data import CacheDataset, DataLoader, decollate_batch, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from skimage.morphology import binary_erosion, binary_dilation, binary_closing
from monai.transforms import MapTransform
from monai.losses import TverskyLoss
from monai.inferers import sliding_window_inference
from scipy.ndimage import label, measurements
import torch.nn as nn


class FocalTverskyLoss(torch.nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, gamma=0.75):
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, y_pred, y_true):
        # Sigmoid uygulama
        y_pred = torch.sigmoid(y_pred)

        # Tversky loss hesaplama
        tp = (y_true * y_pred).sum(dim=(2, 3, 4))
        fn = (y_true * (1 - y_pred)).sum(dim=(2, 3, 4))
        fp = ((1 - y_true) * y_pred).sum(dim=(2, 3, 4))

        tversky_index = tp / (tp + self.alpha * fn + self.beta * fp)

        loss = (1 - tversky_index).pow(self.gamma)

        return loss.mean()


class PreprocessDatad(MapTransform):
    def __init__(self, keys, divider):
        super().__init__(keys)
        self.divider = divider
        
    def __call__(self, x):
        for key in self.keys:
            x[key] = x[key].unsqueeze(0)
            remainder = x[key].shape[3] % self.divider
            if remainder != 0:
                _,H,W,_ = x[key].shape # 1,H,W,D
                x[key] = torch.cat([x[key],torch.zeros(1,H,W,self.divider - remainder)],dim=3)
        return x


class WindowLevelTransformd(MapTransform):
    def __init__(self, keys, window_level=2466, window_width=2797):
        super().__init__(keys)
        self.window_level = window_level
        self.window_width = window_width

    def __call__(self, data):
        for key in self.keys:
            # Apply the window-level transformation
            data[key] = (data[key] - (self.window_level - self.window_width / 2)) / self.window_width
        return data

def apply_morphological_operations(pred, operation='closing'):
    pred_np = pred.cpu().numpy()
    if operation == 'erosion':
        pred_np = binary_erosion(pred_np).astype(np.float32)
    elif operation == 'dilation':
        pred_np = binary_dilation(pred_np).astype(np.float32)
    elif operation == 'closing':
        pred_np = binary_closing(pred_np).astype(np.float32)
    return torch.tensor(pred_np, dtype=torch.float32, device=pred.device)


with open('./train_dataset.json', 'r') as train_js_file:
    train_json_object = json.load(train_js_file)

with open('./val_dataset.json', 'r') as val_js_file:
    val_json_object = json.load(val_js_file)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

train_transform = Compose([
    LoadImaged(keys=["image", "zmap", "label"]),
    PreprocessDatad(keys=["image", "zmap", "label"], divider=32),
    WindowLevelTransformd(keys=["image"], window_level=1090, window_width=262),
    Rand3DElasticd(keys=["image", "zmap", "label"], sigma_range=(5, 8), magnitude_range=(0, 0.03), prob=0.2),
    RandZoomd(keys=["image", "zmap", "label"], min_zoom=0.9, max_zoom=1.1, prob=0.3),
    RandRotated(keys=["image", "zmap", "label"], range_x=1.5, range_y=1.5, range_z=1.5, prob=0.3),
    RandFlipd(keys=["image", "zmap", "label"], prob=0.3, spatial_axis=0),
    RandFlipd(keys=["image", "zmap", "label"], prob=0.3, spatial_axis=1),
    RandFlipd(keys=["image", "zmap", "label"], prob=0.3, spatial_axis=2),
    NormalizeIntensityd(keys=["image", "zmap"], nonzero=True),
    ToTensord(keys=["image", "zmap", "label"])
])

val_transform = Compose([
    LoadImaged(keys=["image", "zmap", "label"]),
    PreprocessDatad(keys=["image", "zmap", "label"], divider=32),
    WindowLevelTransformd(keys=["image"], window_level=1090, window_width=262),
    NormalizeIntensityd(keys=["image", "zmap"], nonzero=True),
    ToTensord(keys=["image", "zmap", "label"])
])


def remove_padding(pred, original_shape):
    _, H, W, D = original_shape
    return pred[:, :H, :W, :D]


def validate_epoch(model, val_loader, dice_metric, device):
    model.eval()
    val_dice_scores = []
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])  # Sigmoid + Thresholding

    with torch.no_grad():
        for val_data in val_loader:
            val_inputs = torch.cat([val_data["image"].to(device), val_data["zmap"].to(device)], dim=1)
            val_labels = val_data["label"].to(device)                

            if val_inputs.shape[4] > 32:
                rand_start = np.random.randint(0, val_inputs.shape[4] - 32)
                val_inputs = val_inputs[:, :, :, :, rand_start:(rand_start + 32)]
                val_labels = val_labels[:, :, :, :, rand_start:(rand_start + 32)]

            roi_size = (128, 128, 32)
            sw_batch_size = 4
            val_outputs = sliding_window_inference(val_inputs, roi_size, sw_batch_size, model)

            for i in range(len(val_outputs)):
                val_outputs[i] = remove_padding(val_outputs[i], val_data["image"][i].shape)

            val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]

            # val_outputs_acc = [apply_connected_component_analysis(i, min_size=500) for i in val_outputs]

            val_labels_decollated = decollate_batch(val_labels)
            dice_score = dice_metric(val_outputs, val_labels_decollated)
            val_dice_scores.append(dice_score.item())

    avg_val_dice = np.mean(val_dice_scores)
    return avg_val_dice


train_dataset = Dataset(train_json_object['training'], transform=train_transform)
val_dataset = Dataset(val_json_object['validation'], transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)

model = SwinUNETR(
    img_size=(128, 128, 128),
    in_channels=2,
    out_channels=1,
    feature_size=48,
    use_checkpoint=True,
    spatial_dims=3
).to(device)

class SwinUNETRWithDropout(nn.Module):
    def __init__(self, model, dropout_rate=0.1):
        super(SwinUNETRWithDropout, self).__init__()
        self.model = model
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.model(x)
        x = self.dropout(x)
        return x

model = SwinUNETRWithDropout(model, dropout_rate=0.1).to(device)

# loss_function = FocalTverskyLoss(alpha=0.5, beta=0.5, gamma=1.33)
loss_function = FocalTverskyLoss(alpha=0.8, beta=0.2, gamma=0.75) # üsttekiydi değişti


optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
lr_scheduler = CosineAnnealingLR(optimizer, T_max=100)

post_trans = Compose([
    Activations(sigmoid=True),
    AsDiscrete(threshold=0.5)
])

dice_metric = DiceMetric(include_background=True, reduction="mean")

from tqdm import tqdm
import time

best_val_dice = 0.0
best_model_path = './best_model.pth'
max_epochs = 500
val_interval = 5
total_start_time = time.time()

for epoch in range(max_epochs):

    epoch_start_time = time.time()
    epoch_loss = 0
    step = 0

    num_batches = len(train_loader)
    progress_bar = tqdm(train_loader, total=num_batches, desc=f"Epoch {epoch + 1}/{max_epochs}", unit="batch")

    for batch_data in progress_bar:
        step += 1

        inputs = torch.cat((batch_data["image"], batch_data["zmap"]), dim=1).to(device)
        labels = batch_data["label"].to(device)


        if inputs.shape[4] > 32:

            rand_start = np.random.randint(0,inputs.shape[4]-32)

            inputs = inputs[:,:,:,:,rand_start:(rand_start+32)]
            labels = labels[:,:,:,:,rand_start:(rand_start+32)]

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

        elapsed_time = time.time() - epoch_start_time
        estimated_total_time = (elapsed_time / step) * num_batches
        remaining_time = estimated_total_time - elapsed_time

        progress_bar.set_postfix({
            "Train Loss": f"{epoch_loss / step:.4f}",
            "Elapsed Time": f"{int(elapsed_time)}s",
            "Remaining Time": f"{int(remaining_time)}s"
        })

        del inputs, labels, loss

    progress_bar.close()

    epoch_end_time = time.time()
    epoch_duration = epoch_end_time - epoch_start_time

    lr_scheduler.step()

    if (epoch + 1) % val_interval == 0:
        avg_val_dice = validate_epoch(model, val_loader, dice_metric, device)
        print(f"Validation Dice Score: {avg_val_dice:.4f}")

        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice

            # Save the model checkpoint
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved with Dice score: {best_val_dice:.4f}")

total_training_time = time.time() - total_start_time
print(f"Total training time: {total_training_time / 60:.2f} minutes.")

