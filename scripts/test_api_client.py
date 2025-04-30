import os
import sys
import requests
from PIL import Image

# URL of the FastAPI server
url = "http://localhost:8000/predict/"

# Path to test image
file_path = "./data/test/sample_input.jpeg"

# Create output directory if it doesn't exist
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

# Derive output filenames based on input filename
basename = os.path.splitext(os.path.basename(file_path))[0]
mask_filename = f"{basename}_predicted_mask.png"
comparison_filename = f"{basename}_side_by_side.png"

mask_path = os.path.join(output_dir, mask_filename)
comparison_path = os.path.join(output_dir, comparison_filename)

# Send image to API
with open(file_path, "rb") as f:
    files = {"file": (file_path, f, "image/png")}
    response = requests.post(url, files=files)

# Check if request was successful
if response.status_code != 200:
    print(f"Request failed with status {response.status_code}: {response.text}")
    sys.exit(1)

# Save the predicted mask
with open(mask_path, "wb") as f:
    f.write(response.content)

print(f"Saved predicted mask as {mask_path}")

# Create side-by-side comparison
try:
    # Load images
    original_image = Image.open(file_path).convert("RGB")
    predicted_mask = Image.open(mask_path).convert("RGB")

    # Resize predicted mask if needed
    if original_image.size != predicted_mask.size:
        predicted_mask = predicted_mask.resize(original_image.size)

    # Create side-by-side image
    side_by_side = Image.new("RGB", (original_image.width * 2, original_image.height))
    side_by_side.paste(original_image, (0, 0))
    side_by_side.paste(predicted_mask, (original_image.width, 0))

    # Save side-by-side
    side_by_side.save(comparison_path)
    print(f"Saved side-by-side comparison as {comparison_path}")

except Exception as e:
    print(f"Failed to create side-by-side comparison: {e}")
    sys.exit(1)
