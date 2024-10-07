import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
from tensorflow.keras.applications import VGG16

# Function to check if an image is empty based on brightness
def is_empty_image(image_path, brightness_threshold=50):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
    avg_brightness = np.mean(img)
    return avg_brightness < brightness_threshold

# Function to calculate entropy of an image
def calculate_entropy(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    hist, _ = np.histogram(img, bins=256, range=(0, 256))
    hist = hist / hist.sum()  # Normalize
    hist = hist[hist > 0]  # Ignore zero entries
    return -np.sum(hist * np.log2(hist))

# Function to extract features from an image
def extract_features(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    mean_intensity = np.mean(img)
    std_dev = np.std(img)
    entropy = calculate_entropy(image_path)
    return [mean_intensity, std_dev, entropy]

# Function to detect outliers using IQR
def detect_outliers_iqr(data, threshold=1.5):
    Q1 = np.percentile(data, 25, axis=0)
    Q3 = np.percentile(data, 75, axis=0)
    IQR = Q3 - Q1
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    outlier_indices = []

    for i in range(data.shape[0]):
        if np.any(data[i] < lower_bound) or np.any(data[i] > upper_bound):
            outlier_indices.append(i)
    return outlier_indices

# Function to build U-Net with VGG16
def build_unet_with_vgg16(input_shape):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    
    # Encoder
    c1 = base_model.get_layer("block1_conv2").output
    c2 = base_model.get_layer("block2_conv2").output
    c3 = base_model.get_layer("block3_conv3").output
    c4 = base_model.get_layer("block4_conv3").output
    c5 = base_model.get_layer("block5_conv3").output

    # Bottleneck
    bottleneck = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c5)

    # Decoder
    u6 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(bottleneck)
    u6 = layers.concatenate([u6, c4])
    c6 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(u6)

    u7 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = layers.concatenate([u7, c3])
    c7 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u7)

    u8 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = layers.concatenate([u8, c2])
    c8 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u8)

    u9 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = layers.concatenate([u9, c1])
    c9 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(u9)

    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = models.Model(inputs=[base_model.input], outputs=[outputs])
    return model

# Preprocess image for prediction
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))  # Resize to match model input
    img_array = img_to_array(img) / 255.0  # Scale pixel values to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Function to predict and save mask
def predict_image(file_path, out_path, model):
    try:
        img_array = preprocess_image(file_path)
        prediction = model.predict(img_array, verbose=False)  # Get the model's predictions
        mask = prediction.squeeze()  # Remove batch dimension

        # Check the shape of the mask
        if mask.ndim == 2:
            mask = np.expand_dims(mask, axis=-1)  # Add a channel dimension if needed

        mask_img = array_to_img(mask)  # Convert array to image
        mask_img.save(out_path)  # Save the mask image
        print(f"Saved mask to {out_path}")
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

# Main function to clean the data and perform segmentation
def clean_image_data(train_data_dir, cleaned_data_dir, model, keep_outliers=True, brightness_threshold=50, outlier_threshold=1.5):
    if not os.path.exists(cleaned_data_dir):
        os.makedirs(cleaned_data_dir)

    features_list = []
    image_paths = []

    # Loop through the files to extract features and remove empty images
    for root, dirs, files in os.walk(train_data_dir):
        for file in tqdm(files, desc=f'Processing files in {os.path.basename(root)}', unit='file'):
            if file.endswith('.jpg'):
                in_path = os.path.join(root, file)

                # Check for empty images
                if not is_empty_image(in_path, brightness_threshold):
                    features = extract_features(in_path)
                    features_list.append(features)
                    image_paths.append(in_path)

    # Convert features list to a NumPy array for analysis
    features_array = np.array(features_list)

    # Detect outliers
    outlier_indices = detect_outliers_iqr(features_array, outlier_threshold)
    print(f"Number of detected outliers: {len(outlier_indices)}")

    # Create a mapping of outlier indices for easier access
    outlier_set = set(outlier_indices)

    # Ask the user whether to save segmentation masks
    save_masks_input = input("Do you want to save segmentation masks? (yes/no): ").strip().lower()
    save_masks = save_masks_input == 'yes'

    # Save cleaned images with progress tracking and perform segmentation
    for i, image_path in enumerate(tqdm(image_paths, desc='Saving cleaned images', unit='file')):
        if keep_outliers or i not in outlier_set:  # Check if outliers should be kept
            # Determine the corresponding cleaned data subdirectory
            relative_path = os.path.relpath(image_path, train_data_dir)
            cleaned_image_path = os.path.join(cleaned_data_dir, relative_path)

            # Create necessary subdirectories in cleaned data folder
            os.makedirs(os.path.dirname(cleaned_image_path), exist_ok=True)

            # Copy the image to the cleaned data directory
            cv2.imwrite(cleaned_image_path, cv2.imread(image_path))

            # Generate and save the segmentation mask if the user opted to save them
            if save_masks:
                mask_output_path = cleaned_image_path.replace('.jpg', '_mask.png')  # Save mask with a different extension
                predict_image(image_path, mask_output_path, model)

# Example usage
if __name__ == "__main__":
    # Get user inputs for training and cleaned data paths
    train_data_dir = input("Enter the path for training data directory: ").strip()
    cleaned_data_dir = input("Enter the path for cleaned data directory: ").strip()

    # Build the U-Net model with VGG16
    unet_model = build_unet_with_vgg16((224, 224, 3))
    unet_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Ask the user whether to keep outliers
    keep_outliers_input = input("Do you want to keep outliers in the dataset? (yes/no): ").strip().lower()
    keep_outliers = keep_outliers_input == 'yes'

    # Run the cleaning and segmentation process
    clean_image_data(train_data_dir, cleaned_data_dir, unet_model, keep_outliers=keep_outliers, brightness_threshold=50, outlier_threshold=1.5)
