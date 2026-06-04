# 2D-to-3D Scene Transformer

This mini-project contains a TensorFlow/Keras transformer that learns to predict a coarse 3D voxel scene from a single 2D front-view image.


namings: 

patch : 
segment of 2D pictures pixels

voxel : 
3D grid of a 3D scene
Each voxel represents a small volume element:

it can store a value like occupancy (0 or 1),
density,
color,
or other 3D information.

## Project overview

The core script is `scene_transformer.py`.
It generates synthetic training data, builds a Vision Transformer-style encoder, and trains a model to predict a voxel grid from a rendered 2D image.

### Key steps in the process

1. Generate a random voxel scene with a few solid blocks.
2. Render the voxel scene from the front into a 2D image with silhouette and depth channels.
3. Split the image into patches and encode each patch into a learned embedding.
4. Apply transformer encoder layers over the patch tokens.
5. Use the special classification token (`cls_token`) as a scene representation.
6. Decode that scene representation into a flattened voxel grid and reshape it to `(voxel_dim, voxel_dim, voxel_dim)`.

## Model architecture

The model built by `build_image_to_scene_model()` contains:

- `Rescaling(1.0 / 255.0)` to normalize image input.
- `PatchEncoder`:
  - Converts the input image into non-overlapping patches.
  - Projects each patch into a fixed embedding dimension.
  - Adds a learnable `cls_token` and positional embeddings.
- A stack of transformer encoder blocks:
  - `LayerNormalization`
  - `MultiHeadAttention` self-attention
  - residual connection
  - feed-forward MLP with `gelu` activation and dropout
  - second residual connection
- A final `LayerNormalization` and token selection:
  - extracts the `cls_token` embedding from position 0
  - passes it through dense layers to predict voxel occupancy
- Output reshaped to a 3D voxel tensor of size `(voxel_dim, voxel_dim, voxel_dim)`.

## Synthetic dataset generation

The training pipeline uses these helper functions:

- `random_voxel_scene(voxel_dim, min_blocks, max_blocks)`
  - Builds a sparse 3D grid of randomly placed rectangular blocks.
- `render_scene_from_front(voxel_scene, image_size)`
  - Renders the front silhouette and depth map from the voxel scene.
  - Produces a 3-channel image for training.
- `build_synthetic_dataset(num_examples, image_size, voxel_dim)`
  - Creates paired input images and target voxel grids.

## Training

The `train()` function:

- builds the model using the selected hyperparameters
- compiles with `Adam` optimizer and `BinaryCrossentropy`
- tracks `binary_accuracy`
- uses `ModelCheckpoint` to save the best model to `./output/transformer_3d_scene.h5`
- uses `EarlyStopping` with `patience=3`

## Command-line usage

Run training from the repository root:

```bash
python scene_transformer.py
```

Available command-line options:

- `--image_size`: input image height/width (default `128`)
- `--patch_size`: patch width/height for patch embeddings (default `16`)
- `--embed_dim`: transformer token embedding size (default `128`)
- `--num_heads`: number of attention heads (default `4`)
- `--mlp_dim`: feed-forward hidden size (default `256`)
- `--num_layers`: number of transformer encoder blocks (default `4`)
- `--voxel_dim`: output voxel grid size (default `32`)
- `--batch_size`: training batch size (default `16`)
- `--epochs`: number of training epochs (default `12`)
- `--train_examples`: number of training examples (default `800`)
- `--val_examples`: number of validation examples (default `200`)
- `--output_dir`: output directory for saved model and checkpoints (default `./output`)
- `--save_weights_csv`: write model weights to CSV after training
- `--csv_dir`: directory for CSV weight files (default `./output/weights_csv`)

Example with CSV export:

```bash
python scene_transformer.py --save_weights_csv --csv_dir ./output/weights_csv
```

## Testing the model

After training, use `test_model.py` to run inference on a folder of images:

```bash
python test_model.py --input_folder ./test_images --output_folder ./test_output
```

Command-line options:

- `--model_path`: path to the saved model (default `./output/transformer_3d_scene.h5`)
- `--input_folder`: folder containing test images (default `./test_images`)
- `--output_folder`: folder to save predictions (default `./test_output`)
- `--image_size`: input image size (must match training size, default `128`)

For each input image, the script produces:

- `{image_name}.npy` — the predicted voxel grid as a NumPy file
- `{image_name}_info.txt` — summary statistics (occupancy range, count of occupied voxels)

## Output files

- `./output/transformer_3d_scene.keras` — best model checkpoint saved during training (Keras format).
- `./output/weights_csv/*` — optional exported weight files if `--save_weights_csv` is used.
- `./test_output/*.npy` — predicted voxel grids from `test_model.py`
- `./test_output/*_info.txt` — prediction statistics from `test_model.py`

## Dependencies

Install the required packages from `requirements.txt`:

```bash
pip install -r requirements.txt
```

If you prefer manual install, the main dependencies are:

```bash
pip install tensorflow scipy numpy
```

## Notes

- This project is intended as a minimal proof of concept for 2D-to-3D scene prediction.
- The synthetic training data is simple and blocky, so the model is not ready for realistic scenes.
- To extend this work, add real 3D data, improve the decoder, or use richer 3D representations such as point clouds or neural radiance fields.
