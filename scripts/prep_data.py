# scripts/prep_data.py

import os

import cv2
import imageio
import nibabel as nib
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def normalize_ct(image, clip_min=-200, clip_max=250):
    """Clip and normalize CT image to 0-255 uint8."""
    image = np.clip(image, clip_min, clip_max)
    image = (image - clip_min) / (clip_max - clip_min) * 255.0
    return image.astype(np.uint8)


def resize_image_and_mask(image_slice, mask_slice, size=(256, 256)):
    """Resize image and mask slices to target size."""
    resized_img = cv2.resize(image_slice, size, interpolation=cv2.INTER_LINEAR)
    resized_mask = cv2.resize(
        mask_slice, size, interpolation=cv2.INTER_NEAREST
    )  # nearest for discrete labels
    return resized_img, resized_mask


def split_patient_ids(patient_ids, seed=42):
    """Split patient IDs into train, val, test sets."""
    train_ids, temp_ids = train_test_split(
        patient_ids, test_size=0.2, random_state=seed
    )
    val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=seed)
    return train_ids, val_ids, test_ids


def prepare_slices(
    images_dir, labels_dir, output_base_dir="data/processed/", size=(256, 256)
):
    """Main function to prepare slices."""

    # Create output directories
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(output_base_dir, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_base_dir, split, "masks"), exist_ok=True)

    # List all patient files
    image_files = sorted([f for f in os.listdir(images_dir) if not f.startswith("._")])
    label_files = sorted([f for f in os.listdir(labels_dir) if not f.startswith("._")])

    assert len(image_files) == len(label_files), "Mismatch between images and labels"

    # Derive patient IDs
    patient_ids = [f.replace(".nii.gz", "") for f in image_files]

    # Split patients
    train_ids, val_ids, test_ids = split_patient_ids(patient_ids)

    # Processing
    for img_file, lbl_file in tqdm(
        zip(image_files, label_files),
        total=len(image_files),
        desc="Processing patients",
    ):
        patient_id = img_file.replace(".nii.gz", "")

        # Determine split
        if patient_id in train_ids:
            split = "train"
        elif patient_id in val_ids:
            split = "val"
        elif patient_id in test_ids:
            split = "test"
        else:
            raise ValueError(f"Patient ID {patient_id} not found in any split!")

        # Load NIfTI volumes
        img_path = os.path.join(images_dir, img_file)
        lbl_path = os.path.join(labels_dir, lbl_file)

        img_nib = nib.load(img_path)
        lbl_nib = nib.load(lbl_path)

        img = img_nib.get_fdata()
        lbl = lbl_nib.get_fdata()

        # Normalize CT scan
        img = normalize_ct(img)

        # Process slice by slice
        for idx in range(img.shape[2]):
            img_slice = img[:, :, idx]
            lbl_slice = lbl[:, :, idx]

            # Skip blank slices (no liver or tumor)
            if not np.any(lbl_slice == 1) and not np.any(lbl_slice == 2):
                continue

            # Resize
            img_slice_resized, lbl_slice_resized = resize_image_and_mask(
                img_slice, lbl_slice, size=size
            )

            # Save
            img_filename = f"{patient_id}_slice{idx:03d}.png"
            lbl_filename = f"{patient_id}_slice{idx:03d}_mask.png"

            imageio.imwrite(
                os.path.join(output_base_dir, split, "images", img_filename),
                img_slice_resized,
            )
            imageio.imwrite(
                os.path.join(output_base_dir, split, "masks", lbl_filename),
                lbl_slice_resized.astype(np.uint8),
            )


if __name__ == "__main__":
    prepare_slices(
        images_dir="data/raw/Task03_Liver/imagesTr/",
        labels_dir="data/raw/Task03_Liver/labelsTr/",
        output_base_dir="data/processed/",
        size=(256, 256),
    )
