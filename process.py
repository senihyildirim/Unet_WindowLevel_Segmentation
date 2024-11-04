from pathlib import Path
import json
from glob import glob
import SimpleITK
import numpy as np
import torch
import os
import torch.nn as nn
from monai.transforms import MapTransform
from monai.networks.nets import SwinUNETR

INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
MODEL1_PATH = Path("model1.pth")
MODEL2_PATH = Path("model2.pth")
MODEL3_PATH = Path("model3.pth")

def get_default_device():
    if torch.cuda.is_available():
        print("Using GPU device")
        return torch.device('cuda')
    else:
        print("Using CPU device")
        return torch.device('cpu')

class PreprocessDatad(MapTransform):
    def __init__(self, keys, divider):
        super().__init__(keys)
        self.divider = divider

    def __call__(self, x):
        for key in self.keys:
            x[key] = torch.tensor(x[key], dtype=torch.float32).unsqueeze(0)
            remainder = x[key].shape[3] % self.divider
            if remainder != 0:
                _, H, W, D = x[key].shape  # 1, H, W, D
                pad_depth = self.divider - remainder
                x[key] = torch.cat([x[key], torch.zeros(1, H, W, pad_depth)], dim=3)
        return x

class BaseNet(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        self.model = SwinUNETR(
            img_size=(128, 128, 128),
            in_channels=2,
            out_channels=1,
            feature_size=48,
            use_checkpoint=True,
            spatial_dims=3
        )
        self.model.load_state_dict(torch.load(model_path, map_location=get_default_device()))
        self.model.eval()


    def forward(self, x):
        x = self.model(x)
        return x


class WindowLevelTransformd(MapTransform):
    def __init__(self, keys, window_level=1090, window_width=262):
        super().__init__(keys)
        self.window_level = window_level
        self.window_width = window_width

    def __call__(self, data):
        for key in self.keys:
            data[key] = (data[key] - (self.window_level - self.window_width / 2)) / self.window_width
        return data

def preprocess_inputs(image, z_map, divider=32):
    # Record the original dimensions before any transformation

    original_shape = image.shape
    
    preprocess_transform = PreprocessDatad(keys=["image", "z_map"], divider=divider)
    # window_level_transform = WindowLevelTransformd(keys=["image"], window_level=1090, window_width=262)

    # Prepare the data dictionary
    data_dict = {"image": image, "z_map": z_map}
    
    # Apply the preprocess transformations
    preprocessed = preprocess_transform(data_dict)
    image_tensor = preprocessed["image"]
    zmap_tensor = preprocessed["z_map"]
    
    # Apply window-level transformation
    # transformed = window_level_transform({"image": image_tensor})
    # image_tensor = transformed["image"]

    
    # Create new tensors if height or width dimensions are less than 32
    if image_tensor.shape[2] < 32 or image_tensor.shape[3] < 32:
        new_height = max(32, image_tensor.shape[2])
        new_width = max(32, image_tensor.shape[3])
        new_image_tensor = torch.zeros((image_tensor.shape[0], image_tensor.shape[1], new_height, new_width), dtype=image_tensor.dtype)
        new_image_tensor[:, :, :image_tensor.shape[2], :image_tensor.shape[3]] = image_tensor
        image_tensor = new_image_tensor

    if zmap_tensor.shape[2] < 32 or zmap_tensor.shape[3] < 32:
        new_height = max(32, zmap_tensor.shape[2])
        new_width = max(32, zmap_tensor.shape[3])
        new_zmap_tensor = torch.zeros((zmap_tensor.shape[0], zmap_tensor.shape[1], new_height, new_width), dtype=zmap_tensor.dtype)
        new_zmap_tensor[:, :, :zmap_tensor.shape[2], :zmap_tensor.shape[3]] = zmap_tensor
        zmap_tensor = new_zmap_tensor

    # Determine padding amounts to make dimensions divisible by divider (e.g., 32)
    pad_depth = (divider - image_tensor.shape[1] % divider) % divider
    pad_height = (divider - image_tensor.shape[2] % divider) % divider
    pad_width = (divider - image_tensor.shape[3] % divider) % divider

    # Apply padding to image and zmap tensors
    image_tensor = torch.nn.functional.pad(image_tensor, (0, pad_width, 0, pad_height, 0, pad_depth), mode="constant", value=0)
    zmap_tensor = torch.nn.functional.pad(zmap_tensor, (0, pad_width, 0, pad_height, 0, pad_depth), mode="constant", value=0)


    # Concatenate image and zmap tensors along the channel dimension
    inputs = torch.cat([image_tensor, zmap_tensor], dim=0).unsqueeze(0)

    print(f"Final concatenated input shape: {inputs.shape}")
    return inputs, original_shape


def remove_padding(tensor, original_shape):
    D, H, W = original_shape
    cropped_tensor = tensor[:, :, :D, :H, :W]
    return cropped_tensor

# def run():
#     # Read the input
#     input_skull_stripped_adc = load_image_file_as_array(
#         location=INPUT_PATH / "images/skull-stripped-adc-brain-mri",
#     )
#     z_adc = load_image_file_as_array(
#         location=INPUT_PATH / "images/z-score-adc",
#     )

#     window_level_configs = [
#         {"window_level": 2000, "window_width": 800},
#         {"window_level": 1090, "window_width": 262},
#         {"window_level": 1800, "window_width": 700}
#     ]

#     models = [MODEL1_PATH, MODEL2_PATH, MODEL3_PATH]
    
#     val_outputs_total = torch.zeros_like(input_skull_stripped_adc, dtype=torch.float32)
#     # Process the inputs
#     with torch.no_grad():
#         input_tensor, original_shape = preprocess_inputs(input_skull_stripped_adc, z_adc, divider=32)
#         input_tensor = input_tensor.to(get_default_device())

#         for model in models:
#             model = BaseNet(model)
#             for window_level_config in window_level_configs:
#                 window_level_transform = WindowLevelTransformd(
#                     keys=["image"],
#                     window_level=window_level_config["window_level"],
#                     window_width=window_level_config["window_width"]
#                 )
#                 transformed_inputs = window_level_transform({"image": input_tensor})["image"]
#                 outputs = model(transformed_inputs)
#                 outputs = torch.sigmoid(outputs)
#                 outputs = (outputs > 0.5).float()

#                 val_outputs_total += outputs

#         out_np = val_outputs_total.cpu().numpy()
#         out_np = (out_np >= 2).astype(int)


#         print("Inference complete. Output sum:", np.sum(out_np))

#     # Create a SimpleITK image from the output and save it
#     hie_segmentation = SimpleITK.GetImageFromArray(out_np)
#     save_image(hie_segmentation)
#     return 0


def run():
    # Read the input
    input_skull_stripped_adc = load_image_file_as_array(
        location=INPUT_PATH / "images/skull-stripped-adc-brain-mri"
    )
    z_adc = load_image_file_as_array(
        location=INPUT_PATH / "images/z-score-adc"
    )

    # Process the inputs
    with torch.no_grad():
        aggregated_output = torch.zeros(input_skull_stripped_adc.shape, device=get_default_device())
        inputs, original_shape = preprocess_inputs(input_skull_stripped_adc, z_adc, divider=32)
        inputs = inputs.to(get_default_device())

        window_level_configs = [
            {"window_level": 2000, "window_width": 800},
            {"window_level": 1090, "window_width": 262},
            {"window_level": 1800, "window_width": 700}
        ]

        # Initialize aggregated_output with the correct shape
        print("Inputs shape:", inputs.shape)
        print("Aggregated output shape:", aggregated_output.shape)

        # Load the models
        models = [
            BaseNet(MODEL1_PATH).to(get_default_device()),
            BaseNet(MODEL2_PATH).to(get_default_device()),
            BaseNet(MODEL3_PATH).to(get_default_device())
        ]

        for model in models:
            model.eval()
            for window_level_config in window_level_configs:
                window_level_transform = WindowLevelTransformd(
                    keys=["image"],
                    window_level=window_level_config["window_level"],
                    window_width=window_level_config["window_width"]
                )
                transformed_inputs = window_level_transform({"image": inputs})["image"]
                outputs = model(transformed_inputs)
                print(f"Model output shape: {outputs.shape}")
                outputs = remove_padding(outputs, original_shape)
                print(f"Model output shape after cropping: {outputs.shape}")
                outputs = torch.sigmoid(outputs)
                outputs = ((outputs > 0.5).float()).squeeze()

                # Ensure outputs match the shape of aggregated_output
                # _, _, output_depth, output_height, output_width = outputs.shape
                # _, _, agg_depth, agg_height, agg_width = aggregated_output.shape

                # if output_depth != agg_depth or output_height != agg_height or output_width != agg_width:
                #     outputs = torch.nn.functional.interpolate(
                #         outputs,
                #         size=(agg_depth, agg_height, agg_width),
                #         mode="trilinear",
                #         align_corners=False
                #     )
                aggregated_output += outputs


        print("Aggregated output shape:", aggregated_output.shape)
        final_output = (aggregated_output >= 2).float()  # Adjust threshold as needed
        out_np = final_output.detach().cpu().numpy().squeeze().astype(np.uint8)
        print("Inference complete. Output sum:", np.sum(out_np))

    # Create a SimpleITK image from the output and save it
    hie_segmentation = SimpleITK.GetImageFromArray(out_np)
    save_image(hie_segmentation)
    return 0

def write_json_file(*, location, content):
    with open(location, 'w') as f:
        f.write(json.dumps(content, indent=4))

def save_image(pred_lesion):
    relative_path = "images/hie-lesion-segmentation"
    output_directory = OUTPUT_PATH / relative_path
    output_directory.mkdir(exist_ok=True, parents=True)

    file_save_name = output_directory / "overlay.mha"
    print(file_save_name)

    SimpleITK.WriteImage(pred_lesion, file_save_name)
    check_file = os.path.isfile(file_save_name)
    print("Check file:", check_file)

def load_image_file_as_array(*, location):
    input_files = glob(str(location / "*.mha"))
    print(input_files[0])
    if not input_files:
        raise FileNotFoundError(f"No .mha files found in the specified directory: {location}")
    result = SimpleITK.ReadImage(input_files[0])
    return SimpleITK.GetArrayFromImage(result)

def _show_torch_cuda_info():
    print("=+=" * 10)
    print("Collecting Torch CUDA information")
    print(f"Torch CUDA is available: {(available := torch.cuda.is_available())}")
    if available:
        print(f"\tnumber of devices: {torch.cuda.device_count()}")
        print(f"\tcurrent device: { (current_device := torch.cuda.current_device())}")
        print(f"\tproperties: {torch.cuda.get_device_properties(current_device)}")
    print("=+=" * 10)

if __name__ == "__main__":
    run()