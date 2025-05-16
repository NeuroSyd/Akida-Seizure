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

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Replace "0" with the GPU index you want to use. If cpu, then ""
print("Num GPUs Available:", len(tf.config.experimental.list_physical_devices('GPU')))

import cnn2snn
from cnn2snn import convert, set_akida_version, AkidaVersion

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

x_train, y_train, x_test, y_test, input_channels, n_classes = EEG_generator(batch_size=batch_size)

output_file_fol = "./Results/" + str(batch_size) + 'B-' + str(args.dataset) + "/"


class SaveEpochData(tf.keras.callbacks.Callback):
    def __init__(self, filename='epoch_data.txt'):
        super(SaveEpochData, self).__init__()
        self.filename = filename

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        # Get the exact metric names dynamically
        precision_key = next((k for k in logs.keys() if "precision" in k.lower()), None)
        recall_key = next((k for k in logs.keys() if "recall" in k.lower()), None)
        auc_key = next((k for k in logs.keys() if "auc" in k.lower()), None)

        train_loss = logs.get('loss', 0)
        val_loss = logs.get('val_loss', 0)

        train_precision = logs.get(precision_key, 0) if precision_key else 0
        val_precision = logs.get(f'val_{precision_key}', 0) if precision_key else 0

        train_recall = logs.get(recall_key, 0) if recall_key else 0
        val_recall = logs.get(f'val_{recall_key}', 0) if recall_key else 0

        train_auc = logs.get(auc_key, 0) if auc_key else 0
        val_auc = logs.get(f'val_{auc_key}', 0) if auc_key else 0

        # Write to file
        with open(self.filename, 'a') as f:
            f.write(f"Epoch {epoch + 1}: "
                    f"Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, "
                    f"Train Precision: {train_precision:.4f}, Validation Precision: {val_precision:.4f}, "
                    f"Train Recall: {train_recall:.4f}, Validation Recall: {val_recall:.4f}, "
                    f"Train AUROC: {train_auc:.4f}, Validation AUROC: {val_auc:.4f}\n")

model_keras = keras.models.Sequential([

    # keras.layers.Rescaling(1. / 255, input_shape=(19, 125, 23)),
    keras.layers.Input(shape=(19, 125, 23)),
    keras.layers.Conv2D(filters=16, kernel_size=(19, 3), strides=(1, 1), padding='same'),
    keras.layers.MaxPooling2D(pool_size=(2, 2), padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.ReLU(),  # THIS IS ALSO COMPATIBLE WITH AKIDA.

    keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid'),
    keras.layers.MaxPooling2D(pool_size=(2, 2), padding='valid'),
    keras.layers.BatchNormalization(),
    keras.layers.ReLU(),

    keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same'),
    keras.layers.MaxPooling2D(pool_size=(2, 2), padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.ReLU(),

    keras.layers.Flatten(),
    keras.layers.Dropout(0.5),

    keras.layers.Dense(128),
    keras.layers.Dropout(0.5),

    keras.layers.ReLU(name='lastRelu'),

    keras.layers.Dense(2),
    keras.layers.Activation('softmax'),  # THIS IS OKAY, it can happen here.

], 'ConvEEG')

model_keras.summary()

model_a = model_keras

model_a.compile(loss='categorical_crossentropy',
    optimizer=Adam(learning_rate=5e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08),
    metrics=[tf.keras.metrics.Precision(),
                       tf.keras.metrics.Recall(),
                       tf.keras.metrics.AUC(curve='ROC')])

model_a.load_weights("....h5")

y_train = y_train.astype('uint8')
y_train = to_categorical(y_train, 2)
y_test = y_test.astype('uint8')
y_test = to_categorical(y_test, 2)

results = model_a.evaluate(x_test, y_test)
print(dict(zip(model_a.metrics_names, results)))

model_dir = "...."

os.makedirs(model_dir, exist_ok=True)

save_epoch_data = SaveEpochData(filename='....txt')

model_checkpoint = keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(model_dir, "epoch{epoch}.h5"),
    save_freq='epoch'
)

#NORMAL QUANTIZATION.
quantized_model_1_0 = cnn2snn.quantize(model_a, input_weight_quantization=8, weight_quantization=4, activ_quantization=4)
quantized_model_1_0.summary()
quantized_model_1_0 = cnn2snn.quantize_layer (quantized_model_1_0, target_layer='lastRelu', bitwidth=1)

#quantization aware training? This is what they did in the paper i'm pretty sure.
quantized_model_1_0.compile(loss='categorical_crossentropy',
    optimizer=Adam(learning_rate=5e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08),
    metrics=[tf.keras.metrics.Precision(),
                       tf.keras.metrics.Recall(),
                       tf.keras.metrics.AUC(curve='ROC')])

quantized_model_1_0.fit(x_train, y_train, epochs=100, batch_size=batch_size, validation_data=(x_test,y_test),
                        callbacks=[model_checkpoint, save_epoch_data])

with set_akida_version(AkidaVersion.v1): #initiate the Model for AKIDA.v1 #following 8/4/4 for compatibility with Akida 1
    akida_mod = convert(quantized_model_1_0) #Bring to Akida Mode version 1.0?

print (akida_mod.summary())

