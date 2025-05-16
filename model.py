import keras
from datasets import EEG_generator
from sklearn.metrics import roc_auc_score, precision_score, recall_score, roc_curve, f1_score
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from keras.optimizers import Adam
from sklearn import metrics
import os
import matplotlib.pyplot as plt
import argparse
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
import numpy as np
import math
from tensorflow.keras.models import Model

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Replace "0" with the GPU index you want to use. If cpu, then ""
print("Num GPUs Available:", len(tf.config.experimental.list_physical_devices('GPU')))

parser = argparse.ArgumentParser(description='Sequential Decision Making..')
parser.add_argument('--dataset', type=str,
                    default='TUH',
                    help='path to load the model')
parser.add_argument('--save', type=str, default='./models/',
                    help='path to load the model')
parser.add_argument('--batch', type=int, default= '128',
                    help="number of branches")
args = parser.parse_args()
batch_size = args.batch

x_train, y_train, x_test, y_test, input_channels, n_classes = \
    EEG_generator(batch_size=batch_size)
output_file_fol = "./Results/" + str(batch_size) + 'B-' + str(args.dataset) + "/"


class SaveEpochData(tf.keras.callbacks.Callback):
    def __init__(self, filename='epoch_data.txt'):
        super(SaveEpochData, self).__init__()
        self.filename = filename

    def on_epoch_end(self, epoch, logs=None):
        # Extracting the metrics from the logs dictionary
        train_loss = logs['loss']
        val_loss = logs['val_loss']
        train_precision = logs['precision']
        val_precision = logs['val_precision']
        train_recall = logs['recall']
        val_recall = logs['val_recall']
        train_auc = logs['auc']
        val_auc = logs['val_auc']

        # Write to a txt file after each epoch
        with open(self.filename, 'a') as f:
            f.write(f"Epoch {epoch + 1}: "
                    f"Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, "
                    f"Train Precision: {train_precision:.4f}, Validation Precision: {val_precision:.4f}, "
                    f"Train Recall: {train_recall:.4f}, Validation Recall: {val_recall:.4f}, "
                    f"Train AUROC: {train_auc:.4f}, Validation AUROC: {val_auc:.4f}\n")

model_keras = keras.models.Sequential([

    # keras.layers.Rescaling(1. / 255, input_shape=(19, 125, 23)),
    keras.layers.Input(shape=(19, 125, 23)),
    # keras.layers.Conv2D(filters=16, kernel_size=(19, 3), strides=(1, 2), padding='same'),
    keras.layers.Conv2D(filters=16, kernel_size=(19, 3), strides=(1, 1), padding='same'),
    keras.layers.MaxPooling2D(pool_size=(2, 2), padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.ReLU(),  # THIS IS ALSO COMPATIBLE WITH AKIDA.

    # keras.layers.Conv2D(filters=32, kernel_size=(1, 3), strides=(1, 2), padding='valid'),
    keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid'),
    keras.layers.MaxPooling2D(pool_size=(2, 2), padding='valid'),
    keras.layers.BatchNormalization(),
    keras.layers.ReLU(),

    # keras.layers.Conv2D(filters=64, kernel_size=(1, 3), strides=(1, 2), padding='same'),
    keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same'),
    keras.layers.MaxPooling2D(pool_size=(2, 2), padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.ReLU(),

    keras.layers.Flatten(),
    keras.layers.Dropout(0.5),

    # keras.layers.Dense(256),
    # keras.layers.Dropout(0.5),
    # keras.layers.ReLU(name='lastRelu'),

    keras.layers.Dense(128),
    keras.layers.Dropout(0.5),  #FOR MY PREVIOUS MODEL.
    keras.layers.ReLU(name='lastRelu'),

    keras.layers.Dense(2),
    keras.layers.Activation('softmax'),  # THIS IS OKAY, it can happen here.

], 'YIKAI_EEG')

model_keras.summary()

model_a = model_keras

model_a.compile(loss='categorical_crossentropy',
    optimizer=Adam(learning_rate=5e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08),
    metrics=[tf.keras.metrics.Precision(),
                       tf.keras.metrics.Recall(),
                       tf.keras.metrics.AUC(curve='ROC')])


model_dir = "..."
os.makedirs(model_dir, exist_ok=True)

save_epoch_data = SaveEpochData(filename='.txt')

#####ONLY IF I WANT TO EXTRACT THE FEATURE EXTRACTOR.

model_a.load_weights(".h5")

y_train = y_train.astype('uint8')
y_train = to_categorical(y_train, 2)
y_test = y_test.astype('uint8')
y_test = to_categorical(y_test, 2)


idx = np.where(y_test == 1)[0][10]  # find the first index that is 1. (200 is 0 (check)) (Check for both 200,99,45,1(oKay),)

x_sample = x_test[idx:idx + 1]  # shape (1,19,125,23)
y_sample = y_test[idx]  # one-hot

fig, axes = plt.subplots(5, 4, figsize=(16, 12))
axes = axes.flatten()

for i in range(19):
    data = x_sample[0, i, :, :]  # shape (125, 23) — freq × time
    ax = axes[i]
    im = ax.imshow(data, cmap='viridis', aspect='auto', origin='lower')
    ax.set_title(f"EEG Input — Channel {i}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Freq")

# Hide any unused subplots
for j in range(19, len(axes)):
    fig.delaxes(axes[j])

fig.suptitle("STFT EEG Input per Channel", fontsize=16)
fig.tight_layout(rect=[0, 0, 1, 0.96])
cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
fig.colorbar(im, cax=cbar_ax, label="Amplitude (normalized)")
plt.show()


def generate_cam(model, input_image, class_idx):
    """
    Generate Class Activation Map (CAM) for a given input image and class index.
    """
    # Identify the last Conv2D layer
    last_conv_layer_name = 'conv2d'  # Update based on model.summary() output
    last_conv_layer = model.get_layer(last_conv_layer_name)

    # Build a model to fetch conv layer output and final prediction
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[last_conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(input_image)
        class_output = predictions[:, class_idx]

    # Get gradients of class output w.r.t. conv layer output
    grads = tape.gradient(class_output, conv_output)

    # Compute the mean intensity of the gradient over spatial locations
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2)).numpy()

    # Convert conv_output to numpy
    conv_output = conv_output[0].numpy()

    # Weight the channels by the corresponding pooled gradients
    for i in range(pooled_grads.shape[-1]):
        conv_output[:, :, i] *= pooled_grads[i]

    # Compute the raw CAM
    cam = np.mean(conv_output, axis=-1)

    # Normalize the CAM
    cam = np.maximum(cam, 0)
    cam = cam / np.max(cam) if np.max(cam) != 0 else cam

    return cam

def generate_cam_from_layer(model, input_image, class_idx, layer_name):
    """
    Generate a CAM from a specified convolutional layer.
    """
    conv_layer = model.get_layer(layer_name)

    # Define model that outputs conv layer and final predictions
    grad_model = Model(inputs=model.input, outputs=[conv_layer.output, model.output])

    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(input_image)
        class_output = predictions[:, class_idx]

    # Compute gradients of class output w.r.t. conv feature map
    grads = tape.gradient(class_output, conv_output)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2)).numpy()
    conv_output = conv_output[0].numpy()

    # Weight channels by pooled grads
    for i in range(pooled_grads.shape[-1]):
        conv_output[:, :, i] *= pooled_grads[i]

    cam = np.mean(conv_output, axis=-1)
    cam = np.maximum(cam, 0)
    cam = cam / np.max(cam) if np.max(cam) != 0 else cam

    return cam

def plot_all_layer_cams(model, input_image, class_idx):
    """
    Generate and plot CAMs for all Conv2D layers.
    """
    conv_layers = [layer.name for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)]

    n_layers = len(conv_layers)
    fig, axes = plt.subplots(nrows=1, ncols=n_layers, figsize=(5 * n_layers, 4))

    if n_layers == 1:
        axes = [axes]

    for i, layer_name in enumerate(conv_layers):
        cam = generate_cam_from_layer(model, input_image, class_idx, layer_name)
        im = axes[i].imshow(cam.T, cmap='jet', aspect='auto', origin='lower')
        axes[i].set_title(f"CAM from {layer_name}")
        axes[i].set_xlabel("Width (Time)")
        axes[i].set_ylabel("Height (Freq/Channels)")
        axes[i].tick_params(axis='both', which='both', length=0)
        fig.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
        # axes[i].axis('off')

    plt.tight_layout()
    plt.show()

# Example usage
input_image = np.expand_dims(x_test[0], axis=0)  # shape: (1, 19, 125, 23)
class_idx = 1  # or 1 depending on the class
plot_all_layer_cams(model_keras, input_image, class_idx)
