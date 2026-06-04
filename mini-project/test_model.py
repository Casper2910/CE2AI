import argparse
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras


def load_and_preprocess_image(image_path: str, image_size: int = 128) -> np.ndarray:
    """
    Load an image from disk and preprocess it to match training format.
    
    Args:
        image_path: path to the image file
        image_size: target image size (will be resized to this)
    
    Returns:
        preprocessed image as numpy array of shape (1, image_size, image_size, 3)
    """
    image_data = tf.io.read_file(image_path)
    image = tf.io.decode_image(image_data, channels=3, expand_animations=False)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [image_size, image_size])
    image = tf.expand_dims(image, axis=0)
    return image.numpy()


def save_voxel_grid(voxel_grid: np.ndarray, output_path: str):
    """
    Save a voxel grid as a NumPy file and as a text representation.
    
    Args:
        voxel_grid: numpy array of shape (voxel_dim, voxel_dim, voxel_dim)
        output_path: path to save the output (without extension)
    """
    np.save(f"{output_path}.npy", voxel_grid)
    
    with open(f"{output_path}_info.txt", "w") as f:
        f.write(f"Shape: {voxel_grid.shape}\n")
        f.write(f"Min value: {voxel_grid.min():.4f}\n")
        f.write(f"Max value: {voxel_grid.max():.4f}\n")
        f.write(f"Mean value: {voxel_grid.mean():.4f}\n")
        f.write(f"Occupied voxels: {np.sum(voxel_grid > 0.5)}\n")


def test_model(
    model_path: str = "./output/transformer_3d_scene.keras",
    input_folder: str = "./test_images",
    output_folder: str = "./test_output",
    image_size: int = 128,
):
    """
    Test the trained model on images in a folder.
    
    Args:
        model_path: path to the saved model file
        input_folder: folder containing test images
        output_folder: folder to save predictions
        image_size: input image size
    """
    os.makedirs(output_folder, exist_ok=True)
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
    
    if not os.path.exists(model_path):
        alt_h5 = model_path + ".h5" if not model_path.endswith(".h5") else model_path[:-3]
        if os.path.exists(alt_h5):
            model_path = alt_h5
        else:
            print(f"Error: Model not found at {model_path}")
            return

    print(f"Loading model from {model_path}...")
    try:
        model = keras.models.load_model(model_path, safe_mode=False)
    except TypeError:
        model = keras.models.load_model(model_path)
    print(f"Model loaded. Summary:")
    model.summary()
    
    image_files = [
        f for f in os.listdir(input_folder)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))
    ]
    
    if not image_files:
        print(f"No image files found in {input_folder}")
        return
    
    print(f"\nFound {len(image_files)} images. Testing...")
    
    for idx, filename in enumerate(image_files, 1):
        image_path = os.path.join(input_folder, filename)
        
        try:
            img = load_and_preprocess_image(image_path, image_size)
            prediction = model.predict(img, verbose=0)
            voxel_grid = prediction[0]
            
            output_name = Path(filename).stem
            output_base = os.path.join(output_folder, output_name)
            save_voxel_grid(voxel_grid, output_base)
            
            print(f"[{idx}/{len(image_files)}] {filename}")
            print(f"  → saved to {output_base}.npy and {output_base}_info.txt")
            print(f"     occupancy range: [{voxel_grid.min():.3f}, {voxel_grid.max():.3f}]")
        
        except Exception as e:
            print(f"[{idx}/{len(image_files)}] {filename} - ERROR: {e}")
    
    print(f"\nTesting complete. Results saved to {output_folder}")


def parse_args():
    parser = argparse.ArgumentParser(description="Test the trained 3D scene transformer on a folder of images.")
    parser.add_argument(
        "--model_path",
        type=str,
        default="./output/transformer_3d_scene.keras",
        help="Path to the saved model file",
    )
    parser.add_argument(
        "--input_folder",
        type=str,
        default="./images",
        help="Folder containing test images",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="./test_output",
        help="Folder where predictions will be saved",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=128,
        help="Input image size (should match model training size)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    test_model(
        model_path=args.model_path,
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        image_size=args.image_size,
    )
