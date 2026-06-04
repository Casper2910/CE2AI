import os  # import the operating system helpers
import argparse  # import command-line argument parsing

import numpy as np  # import numpy for array handling
import scipy.ndimage  # import scipy image utilities
import tensorflow as tf  # import TensorFlow core
from tensorflow import keras  # import Keras API from TensorFlow
from tensorflow.keras import layers  # import Keras layers


def build_encoder_block(embed_dim, num_heads, mlp_dim, dropout):
    inputs = keras.Input(shape=(None, embed_dim))  # create input placeholder for encoder block
    normalized = layers.LayerNormalization(epsilon=1e-6)(inputs)  # normalize inputs
    attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim, dropout=dropout)(normalized, normalized)  # self-attention over inputs
    attention_out = layers.Add()([inputs, attention])  # residual connection after attention

    normalized = layers.LayerNormalization(epsilon=1e-6)(attention_out)  # normalize after attention
    mlp = layers.Dense(mlp_dim, activation="gelu")(normalized)  # feed-forward dense layer
    mlp = layers.Dropout(dropout)(mlp)  # dropout after the first dense layer
    mlp = layers.Dense(embed_dim)(mlp)  # project back to embedding dimension
    mlp_out = layers.Add()([attention_out, mlp])  # residual connection after MLP

    return keras.Model(inputs, mlp_out, name="encoder_block")  # return a small encoder block model


def build_decoder_block(embed_dim, num_heads, mlp_dim, dropout):
    target_inputs = keras.Input(shape=(None, embed_dim))  # target sequence input for decoder self-attention
    encoder_outputs = keras.Input(shape=(None, embed_dim))  # encoder outputs as memory for cross-attention

    normalized_target = layers.LayerNormalization(epsilon=1e-6)(target_inputs)  # normalize decoder target inputs
    self_attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim, dropout=dropout)(
        normalized_target, normalized_target
    )  # decoder self-attention
    self_attention_out = layers.Add()([target_inputs, self_attention])  # residual connection for self-attention

    normalized_target = layers.LayerNormalization(epsilon=1e-6)(self_attention_out)  # normalize before cross-attention
    cross_attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim, dropout=dropout)(
        normalized_target, encoder_outputs
    )  # cross-attention with encoder outputs
    cross_attention_out = layers.Add()([self_attention_out, cross_attention])  # residual connection for cross-attention

    normalized_target = layers.LayerNormalization(epsilon=1e-6)(cross_attention_out)  # normalize before MLP
    mlp = layers.Dense(mlp_dim, activation="gelu")(normalized_target)  # feed-forward layer in decoder
    mlp = layers.Dropout(dropout)(mlp)  # dropout after dense layer
    mlp = layers.Dense(embed_dim)(mlp)  # output projection to embedding size
    decoder_output = layers.Add()([cross_attention_out, mlp])  # residual connection after decoder MLP

    return keras.Model([target_inputs, encoder_outputs], decoder_output, name="decoder_block")  # return decoder block model


def build_simple_image_to_scene_transformer(
    image_size=128,
    patch_size=16,
    embed_dim=128,
    num_heads=4,
    mlp_dim=256,
    num_encoder_layers=2,
    num_decoder_layers=2,
    voxel_dim=32,
    dropout=0.1,
    decoder_query_count=1,
):
    image_input = keras.Input(shape=(image_size, image_size, 3), name="image_input")  # model input for RGB image
    normalized_image = layers.Rescaling(1.0 / 255.0)(image_input)  # scale pixel values to [0, 1]

    patch_embeddings = layers.Conv2D(
        filters=embed_dim,
        kernel_size=patch_size,
        strides=patch_size,
        padding="valid",
        name="patch_projection",
    )(normalized_image)  # convert image to patch embeddings via convolution

    num_patches = (image_size // patch_size) ** 2  # compute number of image patches
    patch_embeddings = layers.Reshape((num_patches, embed_dim))(patch_embeddings)  # flatten patches into sequence form

    patch_indices = tf.range(num_patches)  # create patch position indices
    position_embedding = layers.Embedding(num_patches, embed_dim, name="position_embedding")  # positional embedding layer
    patch_embeddings = patch_embeddings + position_embedding(patch_indices)  # add position information to patches

    encoder_output = patch_embeddings  # start encoder with patch embeddings
    encoder_block = build_encoder_block(embed_dim, num_heads, mlp_dim, dropout)  # build a reusable encoder block
    for _ in range(num_encoder_layers):
        encoder_output = encoder_block(encoder_output)  # stack encoder blocks

    decoder_queries = tf.range(decoder_query_count)  # create decoder query indices
    query_embedding = layers.Embedding(decoder_query_count, embed_dim, name="decoder_query_embedding")  # create learnable query embeddings
    decoder_input = query_embedding(decoder_queries)  # map decoder queries to embeddings
    decoder_input = layers.Lambda(
        lambda queries: tf.tile(tf.expand_dims(queries, 0), [tf.shape(image_input)[0], 1, 1])
    )(decoder_input)  # repeat query embeddings for each batch item

    decoder_output = decoder_input  # initialize decoder input sequence
    decoder_block = build_decoder_block(embed_dim, num_heads, mlp_dim, dropout)  # build a reusable decoder block
    for _ in range(num_decoder_layers):
        decoder_output = decoder_block([decoder_output, encoder_output])  # stack decoder blocks with cross-attention

    decoder_output = layers.LayerNormalization(epsilon=1e-6)(decoder_output)  # normalize decoder output
    decoder_output = layers.GlobalAveragePooling1D()(decoder_output)  # pool sequence into single vector
    decoder_output = layers.Dense(mlp_dim, activation="gelu")(decoder_output)  # projection layer before output
    decoder_output = layers.Dropout(dropout)(decoder_output)  # dropout for regularization
    voxel_output = layers.Dense(voxel_dim ** 3, activation="sigmoid", name="voxel_output")(decoder_output)  # predict voxel occupancy values
    voxel_output = layers.Reshape((voxel_dim, voxel_dim, voxel_dim))(voxel_output)  # reshape flat output into 3D voxel grid

    return keras.Model(inputs=image_input, outputs=voxel_output, name="simple_encoder_decoder_transformer")  # return final model


def random_voxel_scene(voxel_dim=32, min_blocks=2, max_blocks=5):
    scene = np.zeros((voxel_dim, voxel_dim, voxel_dim), dtype=np.float32)  # create empty voxel grid
    num_blocks = np.random.randint(min_blocks, max_blocks + 1)  # choose random number of blocks
    for _ in range(num_blocks):
        size_z = np.random.randint(voxel_dim // 8, voxel_dim // 3)  # random block size in z
        size_y = np.random.randint(voxel_dim // 8, voxel_dim // 3)  # random block size in y
        size_x = np.random.randint(voxel_dim // 8, voxel_dim // 3)  # random block size in x
        start_z = np.random.randint(0, voxel_dim - size_z)  # random z start position
        start_y = np.random.randint(0, voxel_dim - size_y)  # random y start position
        start_x = np.random.randint(0, voxel_dim - size_x)  # random x start position
        scene[
            start_z : start_z + size_z,
            start_y : start_y + size_y,
            start_x : start_x + size_x,
        ] = 1.0  # fill block area with ones
    return np.clip(scene, 0.0, 1.0)  # ensure voxel values stay in [0, 1]


def render_scene_from_front(voxel_scene, image_size=128):
    silhouette = np.max(voxel_scene, axis=0).astype(np.float32)  # project voxels into a 2D silhouette image
    depth = np.zeros_like(silhouette, dtype=np.float32)  # initialize depth map
    for y in range(silhouette.shape[0]):
        for x in range(silhouette.shape[1]):
            depth_line = voxel_scene[:, y, x]  # sample along depth ray
            hit = np.argmax(depth_line > 0.0) if np.any(depth_line > 0.0) else -1  # find first occupied voxel
            if hit >= 0:
                depth[y, x] = 1.0 - (hit / voxel_scene.shape[0])  # compute normalized depth
    silhouette = scipy.ndimage.zoom(silhouette, image_size / silhouette.shape[0], order=1)  # resize silhouette image
    depth = scipy.ndimage.zoom(depth, image_size / depth.shape[0], order=1)  # resize depth image
    image = np.stack([silhouette, depth, silhouette], axis=-1)  # combine channels into RGB-like image
    return np.clip(image, 0.0, 1.0)  # clamp pixel values to valid range


def build_synthetic_dataset(num_examples=1000, image_size=128, voxel_dim=32):
    images = np.zeros((num_examples, image_size, image_size, 3), dtype=np.float32)  # allocate image array
    voxels = np.zeros((num_examples, voxel_dim, voxel_dim, voxel_dim), dtype=np.float32)  # allocate voxel array
    for i in range(num_examples):
        scene = random_voxel_scene(voxel_dim=voxel_dim)  # generate a random 3D scene
        images[i] = render_scene_from_front(scene, image_size=image_size)  # render the scene to an image
        voxels[i] = scene  # store the ground truth voxel scene
    return images, voxels  # return dataset arrays


def train(
    image_size=128,
    voxel_dim=32,
    batch_size=16,
    epochs=8,
    train_examples=800,
    val_examples=200,
    output_dir="./output",
):
    model = build_simple_image_to_scene_transformer(
        image_size=image_size,
        patch_size=16,
        embed_dim=128,
        num_heads=4,
        mlp_dim=256,
        num_encoder_layers=2,
        num_decoder_layers=2,
        voxel_dim=voxel_dim,
        dropout=0.1,
        decoder_query_count=1,
    )  # create the model with default transformer settings

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss=keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=[keras.metrics.BinaryAccuracy(name="binary_accuracy")],
    )  # compile model with optimizer, loss, and metric

    images, voxels = build_synthetic_dataset(num_examples=train_examples + val_examples, image_size=image_size, voxel_dim=voxel_dim)  # build training dataset
    dataset = tf.data.Dataset.from_tensor_slices((images, voxels))  # create TensorFlow dataset from arrays
    dataset = dataset.shuffle(buffer_size=train_examples + val_examples, seed=42)  # shuffle dataset
    train_ds = dataset.take(train_examples).batch(batch_size).prefetch(tf.data.AUTOTUNE)  # create training batch dataset
    val_ds = dataset.skip(train_examples).batch(batch_size).prefetch(tf.data.AUTOTUNE)  # create validation batch dataset

    os.makedirs(output_dir, exist_ok=True)  # create output directory if missing
    checkpoint_path = os.path.join(output_dir, "simple_transformer_scene.keras")  # checkpoint file path

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[
            keras.callbacks.ModelCheckpoint(checkpoint_path, save_best_only=True, monitor="val_loss"),
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        ],
    )  # train the model with checkpointing and early stopping

    print(f"Saved best model to {checkpoint_path}")  # print saved model location
    return model  # return trained model


def parse_args():
    parser = argparse.ArgumentParser(description="Train a simple encoder-decoder transformer for 3D scene prediction.")  # parser for CLI arguments
    parser.add_argument("--epochs", type=int, default=8)  # number of training epochs
    parser.add_argument("--batch_size", type=int, default=16)  # training batch size
    parser.add_argument("--output_dir", type=str, default="./output")  # output directory for model files
    return parser.parse_args()  # return parsed arguments


if __name__ == "__main__":
    args = parse_args()  # parse CLI arguments when run as script
    train(epochs=args.epochs, batch_size=args.batch_size, output_dir=args.output_dir)  # start training with parsed values
