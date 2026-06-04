import argparse
import os

import numpy as np
import scipy.ndimage
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def transformer_encoder(x, embed_dim: int, num_heads: int, mlp_dim: int, dropout: float):
    x_norm = layers.LayerNormalization(epsilon=1e-6)(x)
    attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim, dropout=dropout)(
        x_norm, x_norm
    )
    x = layers.Add()([x, attention_output])

    x_norm = layers.LayerNormalization(epsilon=1e-6)(x)
    mlp_output = keras.Sequential(
        [
            layers.Dense(mlp_dim, activation="gelu"),
            layers.Dropout(dropout),
            layers.Dense(embed_dim),
            layers.Dropout(dropout),
        ]
    )(x_norm)
    x = layers.Add()([x, mlp_output])
    return x


def transformer_decoder(
    x,
    encoder_output,
    embed_dim: int,
    num_heads: int,
    mlp_dim: int,
    dropout: float,
):
    x_norm = layers.LayerNormalization(epsilon=1e-6)(x)
    self_attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim, dropout=dropout)(
        x_norm, x_norm
    )
    x = layers.Add()([x, self_attention_output])

    x_norm = layers.LayerNormalization(epsilon=1e-6)(x)
    cross_attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim, dropout=dropout)(
        x_norm, encoder_output
    )
    x = layers.Add()([x, cross_attention_output])

    x_norm = layers.LayerNormalization(epsilon=1e-6)(x)
    mlp_output = keras.Sequential(
        [
            layers.Dense(mlp_dim, activation="gelu"),
            layers.Dropout(dropout),
            layers.Dense(embed_dim),
            layers.Dropout(dropout),
        ]
    )(x_norm)
    x = layers.Add()([x, mlp_output])
    return x


def build_image_to_scene_model(
    image_size: int = 128,
    patch_size: int = 16,
    embed_dim: int = 128,
    num_heads: int = 4,
    mlp_dim: int = 256,
    num_layers: int = 4,
    voxel_dim: int = 32,
    num_decoder_tokens: int = 1,
    dropout: float = 0.1,
):
    inputs = keras.Input(shape=(image_size, image_size, 3), name="image_input")
    x = layers.Rescaling(1.0 / 255.0)(inputs)
    
    # Patch extraction and projection using Conv2D
    x = layers.Conv2D(
        filters=embed_dim,
        kernel_size=patch_size,
        strides=patch_size,
        padding="valid",
        name="patch_projection",
    )(x)
    
    # Reshape to (batch, num_patches, embed_dim)
    num_patches = (image_size // patch_size) ** 2
    x = layers.Reshape((num_patches, embed_dim))(x)
    
    # Add position embeddings using a constant index tensor
    pos_indices = tf.constant(np.arange(num_patches, dtype=np.int32))
    pos_embed = layers.Embedding(num_patches, embed_dim, name="position_embedding")(pos_indices)
    x = layers.Add()([x, pos_embed])

    for _ in range(num_layers):
        x = transformer_encoder(x, embed_dim=embed_dim, num_heads=num_heads, mlp_dim=mlp_dim, dropout=dropout)

    encoder_output = layers.LayerNormalization(epsilon=1e-6)(x)

    decoder_indices = tf.constant(np.arange(num_decoder_tokens, dtype=np.int32))
    decoder_input_embed = layers.Embedding(num_decoder_tokens, embed_dim, name="decoder_input_embedding")(decoder_indices)
    decoder_pos_embed = layers.Embedding(num_decoder_tokens, embed_dim, name="decoder_position_embedding")(decoder_indices)
    decoder_queries = layers.Add()([decoder_input_embed, decoder_pos_embed])
    decoder_inputs = layers.Lambda(
        lambda args: tf.tile(tf.expand_dims(args[0], 0), [tf.shape(args[1])[0], 1, 1]),
        name="decoder_query_tiling",
    )([decoder_queries, inputs])

    x = decoder_inputs
    for _ in range(num_layers):
        x = transformer_decoder(
            x,
            encoder_output=encoder_output,
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            dropout=dropout,
        )

    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(mlp_dim, activation="gelu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(voxel_dim ** 3, activation="sigmoid", name="voxel_output")(x)
    outputs = layers.Reshape((voxel_dim, voxel_dim, voxel_dim))(x)

    return keras.Model(inputs=inputs, outputs=outputs, name="image_to_scene_transformer")


def random_voxel_scene(voxel_dim: int = 32, min_blocks: int = 2, max_blocks: int = 5):
    scene = np.zeros((voxel_dim, voxel_dim, voxel_dim), dtype=np.float32)
    num_blocks = np.random.randint(min_blocks, max_blocks + 1)

    for _ in range(num_blocks):
        size_z = np.random.randint(voxel_dim // 8, voxel_dim // 3)
        size_y = np.random.randint(voxel_dim // 8, voxel_dim // 3)
        size_x = np.random.randint(voxel_dim // 8, voxel_dim // 3)
        start_z = np.random.randint(0, voxel_dim - size_z)
        start_y = np.random.randint(0, voxel_dim - size_y)
        start_x = np.random.randint(0, voxel_dim - size_x)

        scene[
            start_z : start_z + size_z,
            start_y : start_y + size_y,
            start_x : start_x + size_x,
        ] = 1.0

    return np.clip(scene, 0.0, 1.0)


def render_scene_from_front(voxel_scene: np.ndarray, image_size: int = 128) -> np.ndarray:
    silhouette = np.max(voxel_scene, axis=0).astype(np.float32)
    depth = np.zeros_like(silhouette, dtype=np.float32)

    for y in range(silhouette.shape[0]):
        for x in range(silhouette.shape[1]):
            depth_line = voxel_scene[:, y, x]
            hit = np.argmax(depth_line > 0.0) if np.any(depth_line > 0.0) else -1
            if hit >= 0:
                depth[y, x] = 1.0 - (hit / voxel_scene.shape[0])

    silhouette = scipy.ndimage.zoom(silhouette, image_size / silhouette.shape[0], order=1)
    depth = scipy.ndimage.zoom(depth, image_size / depth.shape[0], order=1)
    image = np.stack([silhouette, depth, silhouette], axis=-1)
    return np.clip(image, 0.0, 1.0)


def build_synthetic_dataset(
    num_examples: int = 1000,
    image_size: int = 128,
    voxel_dim: int = 32,
    min_blocks: int = 2,
    max_blocks: int = 5,
):
    images = np.zeros((num_examples, image_size, image_size, 3), dtype=np.float32)
    voxels = np.zeros((num_examples, voxel_dim, voxel_dim, voxel_dim), dtype=np.float32)

    for i in range(num_examples):
        scene = random_voxel_scene(voxel_dim=voxel_dim, min_blocks=min_blocks, max_blocks=max_blocks)
        images[i] = render_scene_from_front(scene, image_size=image_size)
        voxels[i] = scene

    return images, voxels


def save_model_weights_to_csv(model: keras.Model, csv_dir: str = "./output/weights_csv"):
    os.makedirs(csv_dir, exist_ok=True)

    for weight in model.weights:
        safe_name = weight.name.replace("/", "_").replace(":", "_")
        csv_path = os.path.join(csv_dir, f"{safe_name}.csv")
        array = weight.numpy()
        header = f"shape:{array.shape}"

        if array.ndim <= 2:
            np.savetxt(csv_path, array, delimiter=",", header=header, comments="# ")
        else:
            with open(csv_path, "w") as f:
                f.write(f"# {header}\n")
                np.savetxt(f, array.reshape(-1), delimiter=",")

    print(f"Saved model weights to CSV files in {csv_dir}")


def prepare_datasets(
    train_examples: int = 800,
    val_examples: int = 200,
    image_size: int = 128,
    voxel_dim: int = 32,
    batch_size: int = 16,
):
    images, voxels = build_synthetic_dataset(
        num_examples=train_examples + val_examples,
        image_size=image_size,
        voxel_dim=voxel_dim,
    )

    dataset = tf.data.Dataset.from_tensor_slices((images, voxels))
    dataset = dataset.shuffle(buffer_size=train_examples + val_examples, seed=42)
    train_ds = dataset.take(train_examples).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_ds = dataset.skip(train_examples).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return train_ds, val_ds


def train(
    image_size: int = 128,
    patch_size: int = 16,
    embed_dim: int = 128,
    num_heads: int = 4,
    mlp_dim: int = 256,
    num_layers: int = 4,
    voxel_dim: int = 32,
    batch_size: int = 16,
    epochs: int = 12,
    train_examples: int = 800,
    val_examples: int = 200,
    output_dir: str = "./output",
    save_weights_csv: bool = False,
    csv_dir: str = "./output/weights_csv",
):
    model = build_image_to_scene_model(
        image_size=image_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        mlp_dim=mlp_dim,
        num_layers=num_layers,
        voxel_dim=voxel_dim,
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss=keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=[keras.metrics.BinaryAccuracy(name="binary_accuracy")],
    )

    train_ds, val_ds = prepare_datasets(
        train_examples=train_examples,
        val_examples=val_examples,
        image_size=image_size,
        voxel_dim=voxel_dim,
        batch_size=batch_size,
    )

    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = os.path.join(output_dir, "transformer_3d_scene.keras")

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[
            keras.callbacks.ModelCheckpoint(checkpoint_path, save_best_only=True, monitor="val_loss"),
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        ],
    )

    print(f"Saved best model to {checkpoint_path}")
    if save_weights_csv:
        save_model_weights_to_csv(model, csv_dir=csv_dir)
    return model


def parse_args():
    parser = argparse.ArgumentParser(description="Train a transformer for 3D scene prediction from 2D images.")
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--mlp_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--voxel_dim", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--train_examples", type=int, default=800)
    parser.add_argument("--val_examples", type=int, default=200)
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument(
        "--save_weights_csv",
        action="store_true",
        help="Save the trained model weights to CSV files after training.",
    )
    parser.add_argument(
        "--csv_dir",
        type=str,
        default="./output/weights_csv",
        help="Directory where CSV weight files are written.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        image_size=args.image_size,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        mlp_dim=args.mlp_dim,
        num_layers=args.num_layers,
        voxel_dim=args.voxel_dim,
        batch_size=args.batch_size,
        epochs=args.epochs,
        train_examples=args.train_examples,
        val_examples=args.val_examples,
        output_dir=args.output_dir,
        save_weights_csv=args.save_weights_csv,
        csv_dir=args.csv_dir,
    )
