import apify
import cv2
import numpy as np
from segment_anything import sam_model_registry
from PIL import Image

# Initialize SAM model
sam_checkpoint = "path/to/sam_model.pth"
model_type = "vit_h"
sam = sam_model_registry[sam_checkpoint](model_type)

def segment_image(image_path):
    image = cv2.imread(image_path)
    masks = sam.predict(image)
    return masks

def reconstruct_parts(image, masks):
    reconstructed_parts = []
    for mask in masks:
        part = apply_inpainting(image, mask)
        reconstructed_parts.append(part)
    return reconstructed_parts

def apply_inpainting(image, mask):
    inpainted_image = cv2.inpaint(image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    return inpainted_image

# Apify actor main function
async def main():
    # Get input from Apify
    input_data = await apify.get_input()
    image_url = input_data['imageUrl']
    image_path = download_image(image_url)

    # Segment the image
    masks = segment_image(image_path)

    # Reconstruct parts from masks
    image = cv2.imread(image_path)
    parts = reconstruct_parts(image, masks)

    # Save or upload results
    await upload_results(parts)

# Run the Apify actor
if __name__ == '__main__':
    apify.run(main)