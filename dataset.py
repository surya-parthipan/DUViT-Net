import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import nibabel as nib

class MSDDataset(Dataset):
    def __init__(self, image_paths, mask_paths, modalities=None, slice_axis=2, transform=None):
        """
        image_paths: List of paths to the image files.
        mask_paths: List of paths to the mask files.
        modalities: List of indices to select modalities/channels. None means use all available channels.
        slice_axis: Axis along which to take slices (0, 1, or 2).
        transform: Optional transformations to apply.
        """
        super().__init__()
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.modalities = modalities
        self.slice_axis = slice_axis
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image and mask
        img = nib.load(self.image_paths[idx]).get_fdata()
        mask = nib.load(self.mask_paths[idx]).get_fdata()

        # Handle modalities (channels)
        if img.ndim == 4:
            # Multiple modalities
            if self.modalities is not None:
                img = img[..., self.modalities]  # Select specified modalities
            else:
                img = img  # Use all modalities
            channels = img.shape[-1]
        else:
            # Single modality
            img = img[..., np.newaxis]
            channels = 1

        # Normalize image to [0, 1]
        img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)

        # Ensure mask is in appropriate format
        mask = mask.astype(np.int64)  # Use int64 for segmentation masks

        # Extract a slice along the specified axis
        mid_slice = img.shape[self.slice_axis] // 2
        img = np.take(img, mid_slice, axis=self.slice_axis)
        mask = np.take(mask, mid_slice, axis=self.slice_axis)

        # Transpose to [C, H, W]
        img = np.transpose(img, (2, 0, 1)) if channels > 1 else img[np.newaxis, ...]
        mask = mask[np.newaxis, ...]

        # Convert to tensors
        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).long()  # For multi-class segmentation

        # Resize to 256x256
        img = F.interpolate(img.unsqueeze(0), size=(256, 256), mode='bilinear', align_corners=False).squeeze(0)
        mask = F.interpolate(mask.unsqueeze(0).float(), size=(256, 256), mode='nearest').squeeze(0).long()

        if self.transform:
            img = self.transform(img)

        return img, mask


# import torch
# import torch.nn.functional as F
# from torch.utils.data import Dataset
# import numpy as np
# import nibabel as nib

# class MSDDataset(Dataset):
#     def __init__(self, image_paths, mask_paths, modalities=None, slice_axis=2, transform=None):
#         """
#         image_paths: List of paths to the image files.
#         mask_paths: List of paths to the mask files.
#         modalities: List of indices to select modalities/channels. None means use all available channels.
#         slice_axis: Axis along which to take slices (0, 1, or 2).
#         transform: Optional transformations to apply.
#         """
#         super().__init__()
#         self.image_paths = image_paths
#         self.mask_paths = mask_paths
#         self.modalities = modalities
#         self.slice_axis = slice_axis
#         self.transform = transform

#     def __len__(self):
#         return len(self.image_paths)

#     def __getitem__(self, idx):
#         # Load image and mask
#         img = nib.load(self.image_paths[idx]).get_fdata()
#         mask = nib.load(self.mask_paths[idx]).get_fdata()

#         # Handle modalities (channels)
#         if img.ndim == 4:
#             # Multiple modalities
#             if self.modalities is not None:
#                 img = img[..., self.modalities]  # Select specified modalities
#             else:
#                 img = img  # Use all modalities
#             channels = img.shape[-1]
#         else:
#             # Single modality
#             img = img[..., np.newaxis]
#             channels = 1

#         # Normalize image to [0, 1]
#         img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)

#         # Ensure mask is in appropriate format
#         mask = mask.astype(np.int64)  # Use int64 for segmentation masks

#         # Extract a slice along the specified axis
#         mid_slice = img.shape[self.slice_axis] // 2
#         img = np.take(img, mid_slice, axis=self.slice_axis)
#         mask = np.take(mask, mid_slice, axis=self.slice_axis)

#         # Transpose to [C, H, W]
#         img = np.transpose(img, (2, 0, 1)) if channels > 1 else img[np.newaxis, ...]
#         mask = mask[np.newaxis, ...]

#         # Convert to tensors
#         img = torch.from_numpy(img).float()
#         mask = torch.from_numpy(mask).long()  # For multi-class segmentation

#         # Resize to 256x256
#         img = F.interpolate(img.unsqueeze(0), size=(256, 256), mode='bilinear', align_corners=False).squeeze(0)
#         mask = F.interpolate(mask.unsqueeze(0).float(), size=(256, 256), mode='nearest').squeeze(0).long()

#         if self.transform:
#             img = self.transform(img)

#         return img, mask
