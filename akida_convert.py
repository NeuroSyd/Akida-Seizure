import keras
from datasets import EEG_generator, RPA_generator_KERAS, float16_to_uint8, split_dataset_into_shot_and_inference
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
from time import time
from akida import FullyConnected, evaluate_sparsity, Model, AkidaUnsupervised
import os

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
parser.add_argument('--pat_n', type=str, default= '2',
                    help="number of branches")
parser.add_argument('--shots', type=int, default= '1',
                    help="number of branches")

args = parser.parse_args()
batch_size = args.batch

x_train, y_train, x_test, y_test, input_channels, n_classes = EEG_generator(batch_size=batch_size)

y_train = y_train.astype('uint8')
y_train = to_categorical(y_train, 2)

y_test = y_test.astype('uint8')
y_test = to_categorical(y_test, 2)

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
    keras.layers.Conv2D(filters=16, kernel_size=(19, 3), strides=(1,1), padding='same'),
    keras.layers.MaxPooling2D(pool_size=(2, 2), padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.ReLU(),  # THIS IS ALSO COMPATIBLE WITH AKIDA.

    keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding='valid'),
    keras.layers.MaxPooling2D(pool_size=(2, 2), padding='valid'),
    keras.layers.BatchNormalization(),
    keras.layers.ReLU(),

    keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same'),
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

#quantization aware training
quantized_model_1_0 = cnn2snn.quantize(model_a, input_weight_quantization=8, weight_quantization=4, activ_quantization=4)
quantized_model_1_0.summary()
quantized_model_1_0 = cnn2snn.quantize_layer (quantized_model_1_0, target_layer='lastRelu', bitwidth=1)

#Higuest auroc.
quantized_model_1_0.load_weights(".h5")

quantized_model_1_0.compile(loss='categorical_crossentropy',
    optimizer=Adam(learning_rate=5e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08),
    metrics=[tf.keras.metrics.Precision(),
                       tf.keras.metrics.Recall(),
                       tf.keras.metrics.AUC(curve='ROC')])

results = quantized_model_1_0.evaluate(x_test, y_test)

print(dict(zip(quantized_model_1_0.metrics_names, results))) #GOOD. NO NEED OF QAT

with set_akida_version(AkidaVersion.v1):
    akida_mod = convert(quantized_model_1_0)
akida_mod.summary()

num_samples_to_use = int(len(x_train)*0.2)
x_train = float16_to_uint8(x_train[:num_samples_to_use])

sparsities = evaluate_sparsity(akida_mod, x_train) #compute sparsity.

# Retrieve the number of output spikes from the feature extractor output
output_density = 1 - sparsities[akida_mod.get_layer('conv2d_2')] #LAST LAYER.
avg_spikes = akida_mod.get_layer('conv2d_2').output_dims[-1] * output_density #LAST LAYER
print(f"Average number of spikes: {avg_spikes}")

num_weights = int(1.2 * abs(avg_spikes))
print("The number of weights is then set to:", num_weights)

akida_mod.pop_layer()
akida_mod.summary()

#### JUST FOR MY OWN VALIDATION
from akida import AkidaUnsupervised

# num_neurons_per_class = 2000
num_neurons_per_class = 1500

num_classes = 2

layer_fc = FullyConnected(name='akida_edge_layer',
                          units=num_classes * num_neurons_per_class,
                          activation=False)
akida_mod.add(layer_fc)
akida_mod.compile(optimizer=AkidaUnsupervised(num_weights=num_weights,
                                             num_classes=num_classes,
                                             learning_competition=0.1))

akida_mod.summary()

Rep = 5

path = "..."
#python3.10 akida_convert.py --pat_n 2 --shots 1
shots = args.shots
# patname = args.pat_n

# Targets = [str(i) for i in range(17, 31)]
# '3','5','10','12','25'
Targets = ['5']
for patname in Targets:

    base_dir = f"results/patient_{patname}/shot_{shots}"
    os.makedirs(base_dir, exist_ok=True)
    results_path = os.path.join(base_dir, "metrics.txt")

    for x in range(Rep):
        class_shots, class_inf = split_dataset_into_shot_and_inference(patname=patname, year="..", shot_count=shots)
        akida_rep = akida_mod

        for cls in sorted(class_shots.keys()):
            start = time()
            train_images = class_shots[cls]
            for image in train_images:
                image = np.expand_dims(image, axis=0)
                akida_rep.fit(image, cls)
            end = time()
            print(f'Learned (class {cls}) with {len(train_images)} sample(s) in {end-start:.2f}s')

        import statistics as stat

        def predict_probabilities(outputs, num_classes):
            """Predicts the class probabilities for the specified inputs.

            Args:
                inputs (:obj:`numpy.ndarray`): a (n, x, y, c) uint8 tensor
                num_classes (int, optional): the number of output classes
                batch_size (int, optional): maximum number of inputs that should be
                    processed at a time

            Returns:
                :obj:`numpy.ndarray`: an array of probability scores per class
            """

            # Compute softmax in a numerically stable way:
            # exp_outputs = np.exp(outputs - np.max(outputs, axis=-1, keepdims=True))
            # softmax_outputs = (exp_outputs / np.sum(exp_outputs, axis=-1, keepdims=True)) #(1,1,1,2000)

            exp_outputs = np.exp(outputs - np.max(outputs, axis=-1, keepdims=True))
            softmax_outputs = (exp_outputs / np.sum(exp_outputs, axis=-1, keepdims=True)) #(1,1,1,2000)

            num_neurons = outputs.shape[-1]

            if num_classes != 0 and num_classes != num_neurons:
                neurons_per_class = num_neurons // num_classes
                softmax_outputs = softmax_outputs.reshape((-1, num_classes, neurons_per_class))
                softmax_outputs = np.sum(softmax_outputs, axis=-1)  # Sum activations per class
                if num_classes == 2:
                    return softmax_outputs[..., 1]  # Equivalent to output[:, 1] in PyTorch

            return softmax_outputs # Returns a 1D array of class probabilities


        all_y_true = []
        all_y_scores = []  # For AUROC (continuous probabilities)

        def plot_AUROC_Find_Threshold(y_true, y_scores):
            # Compute global metrics (AUROC, precision, and recall)
            auroc = roc_auc_score(y_true, y_scores)
            fpr, tpr, thresholds = metrics.roc_curve(y_true, y_scores)
            # Create ROC curve plot
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % auroc)
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.0])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC)')
            plt.legend(loc='lower right')
            plt.show()
            print (auroc)

            youden_j = tpr - fpr
            optimal_idx = np.argmax(youden_j)
            optimal_threshold = thresholds[optimal_idx]

            return optimal_threshold, auroc

        for cls in sorted(class_inf.keys()):

            if cls == 0:
                test_images = class_inf[cls]
            else:
                test_images = class_inf[cls]

            predictions = np.zeros(len(test_images))
            scores = np.zeros(len(test_images))  # Store probability scores

            for j in range(len(test_images)):
                image_inf = np.expand_dims(test_images[j], axis=0)
                output = akida_rep.predict(image_inf)
                prob = predict_probabilities(output,num_classes=num_classes) #SHOULD RETURN (1,) as the other one.

                all_y_true.append(cls)
                all_y_scores.append(prob)

        # Convert lists to numpy arrays for metric computation
        y_true = np.array(all_y_true)
        y_scores = np.concatenate(all_y_scores, axis=0)
        threshold, auroc = plot_AUROC_Find_Threshold(y_true, y_scores)
        binary_predictions = (y_scores >= threshold).astype(int)

        cm = confusion_matrix(y_true, binary_predictions)
        tn, fp, fn, tp = cm.ravel()
        # Calculate FPR, Sensitivity, AUCROC
        fpr = fp / (fp + tn)
        sensitivity = tp / (tp + fn)

        print(f'False Positive Rate (FPR): {fpr:.2f}')
        print(f'Sensitivity: {sensitivity:.2f}')
        print (f'For: {shots} shots')

        import seaborn as sns

        # Plot Confusion Matrix
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix")
        plt.show()

        with open(results_path, 'a') as file:
            # Create the formatted string
            output_str = 'auroc: {:.3f}, fpr: {:.3f}, sensitivity: {:.3f}'.\
                format(auroc, fpr, sensitivity)
            # Write the string to the file
            file.write(output_str + '\n')

