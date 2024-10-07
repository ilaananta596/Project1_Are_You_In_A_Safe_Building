import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
import csv
from tqdm import tqdm
import argparse

# Define command-line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description='Train and test model.')
    parser.add_argument('--train_dir', required=True, help='Path to the training data directory')
    parser.add_argument('--test_dir', required=True, help='Path to the test data directory')
    parser.add_argument('--output', default='predictions.csv', help='Path to save predictions CSV')
    return parser.parse_args()

# Data loading and processing
def preprocess_and_convert(dataset):
    images = []
    labels = []
    for image, label in tqdm(dataset, desc="Loading dataset"):
        images.append(image.numpy())
        labels.append(label.numpy())
    return np.array(images).squeeze(), np.array(labels).squeeze()

def preprocess_image_input(input_images, labels):
    return tf.keras.applications.resnet50.preprocess_input(input_images), labels

# Define model
def feature_extractor(inputs):
    feature_extractor = tf.keras.applications.resnet.ResNet50(input_shape=(224, 224, 3),
                                               include_top=False,
                                               weights='imagenet')(inputs)
    return feature_extractor

def classifier(inputs):
    x = tf.keras.layers.GlobalAveragePooling2D()(inputs)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(5, activation="softmax")(x)
    return x

def final_model(inputs):
    resnet_feature_extractor = feature_extractor(inputs)
    classification_output = classifier(resnet_feature_extractor)
    return classification_output

def define_compile_model():
    inputs = tf.keras.layers.Input(shape=(224,224,3))
    classification_output = final_model(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=classification_output)

    model.compile(optimizer='SGD',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Preprocess test images
def preprocess_image(img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

def standardize_image(img_array, scaler):
    img_reshaped = img_array.reshape(-1, 224 * 224 * 3)
    img_standardized = scaler.transform(img_reshaped).reshape(img_array.shape)
    return img_standardized

# Main function
def main():
    print("Parsing arguments...")
    args = parse_arguments()

    # Load train data
    print("Loading training and validation data...")
    train_data_dir = args.train_dir
    test_data_dir = args.test_dir

    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        train_data_dir,
        image_size=(224, 224),
        batch_size=1,
        label_mode='int',
        class_names=['A', 'B', 'C', 'D', 'S'],
        shuffle=True,
        validation_split=0.1,
        subset='training',
        seed=123
    )

    val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        train_data_dir,
        image_size=(224, 224),
        batch_size=1,
        label_mode='int',
        class_names=['A', 'B', 'C', 'D', 'S'],
        shuffle=True,
        validation_split=0.1,
        subset='validation',
        seed=123
    )

    print("Preprocessing and normalizing training and validation data...")
    # Preprocess data
    train_dataset = train_dataset.map(preprocess_image_input)
    val_dataset = val_dataset.map(preprocess_image_input)

    train_X, train_y = preprocess_and_convert(train_dataset)
    val_X, val_y = preprocess_and_convert(val_dataset)

    # Normalize and standardize data
    train_X_normalized = train_X / 255.0
    val_X_normalized = val_X / 255.0

    scaler = StandardScaler()
    train_X_reshaped = train_X_normalized.reshape(-1, 224 * 224 * 3)
    scaler.fit(train_X_reshaped)
    train_X_standardized = scaler.transform(train_X_reshaped).reshape(train_X_normalized.shape)
    val_X_reshaped = val_X_normalized.reshape(-1, 224 * 224 * 3)
    val_X_standardized = scaler.transform(val_X_reshaped).reshape(val_X_normalized.shape)

    # Convert labels to categorical
    training_labels = to_categorical(train_y, 5)
    validation_labels = to_categorical(val_y, 5)

    # Define and compile model
    print("Defining and compiling the model...")
    model = define_compile_model()
    model.summary()

    # Define callbacks
    print("Setting up callbacks...")
    lr_scheduler = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=2, min_lr=1e-6)
    early_stopping = EarlyStopping(monitor='val_accuracy', mode='max', patience=10, restore_best_weights=True)
    callbacks = [lr_scheduler, early_stopping]

    # Data augmentation
    print("Applying data augmentation...")
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Train model
    print("Starting training...")
    model.fit(datagen.flow(train_X_standardized, training_labels, batch_size=16),
              epochs=100,
              validation_data=(val_X_standardized, validation_labels),
              verbose=1,
              callbacks=callbacks)

    # Predict on test data
    print("Making predictions on test data...")
    results = {}
    for root, dirs, files in os.walk(test_data_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                file_path = os.path.join(root, file)
                img_array = preprocess_image(file_path)
                img_standardized = standardize_image(img_array, scaler)
                predictions = model.predict(img_standardized)
                predicted_label = np.argmax(predictions, axis=1)[0]
                results[file] = predicted_label

    print("Saving results to CSV...")
    # Save results to CSV
    results_final = [{'ID': int(file_name.split('.')[0]), 'Predictions': pred + 1} for file_name, pred in results.items()]
    results_final.sort(key=lambda x: x['ID'])

    with open(args.output, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['ID', 'Predictions'])
        writer.writeheader()
        writer.writerows(results_final)

    print(f"Predictions saved to {args.output}")

if __name__ == "__main__":
    main()
