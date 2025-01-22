import os
import argparse
from PIL import Image
import torchvision.transforms as T

def parse_args():
    """
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Resize and center-crop images in a folder to 256x256."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to the input folder containing images."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the output folder to save processed images."
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=256,
        help="Desired size for both resize and center-crop."
    )
    return parser.parse_args()

def main():
    args = parse_args()

    # Define the transform pipeline:
    transform = T.Compose([
        T.Resize(args.img_size),                 # Resize (shortest edge = img_size, keeps aspect ratio)
        T.CenterCrop((args.img_size, args.img_size))  # Center-crop to exactly img_size x img_size
    ])

    # Create the output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Loop over all files in the input directory
    for filename in os.listdir(args.input_dir):
        # Check if the file is an image by extension
        if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
            input_path = os.path.join(args.input_dir, filename)
            
            # Open the image
            img = Image.open(input_path)

            # Apply the transforms
            img_transformed = transform(img)

            # Save the transformed image with the same filename in the output directory
            output_path = os.path.join(args.output_dir, filename)
            img_transformed.save(output_path)
            print(f"Processed and saved: {output_path}")

if __name__ == "__main__":
    main()
