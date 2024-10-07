# Automated Multi-Class Building Classification from Street View Images

## Objective
This project focuses on developing a machine learning model to automatically classify building types from street view images. The goal is to use this classification to assess the seismic vulnerability of buildings, aiding in earthquake risk management. By leveraging image data and machine learning, the model helps identify potentially high-risk areas based on building types.

## Dataset
- **Training Data**: 2,516 street view images, categorized into five classes (A, B, C, D, S), each representing different building materials and structures. Images are labeled as `<img_no>_<class_name>`.
- **Test Data**: 478 images labeled sequentially as `<img_no>`.

## Preprocessing and Outlier Detection
1. **Empty Image Removal**: A function checks for "empty" images based on their brightness and removes those that don’t contribute useful information.
2. **Outlier Detection**: Using the IQR method, images without clear building features (e.g., occluded by trees) are identified and removed.
3. **Scaling**: All images are normalized by dividing pixel values by 255 and standardized using the training set’s mean and standard deviation.

## Data Augmentation
To increase the size and variability of the dataset, data augmentation techniques such as flipping, shifting, shearing, and zooming were applied to the training images.

## Key Files
- **`TrainAndaTestResnet50.py`**: Contains the training pipeline for ResNet50 and outputs the results in CSV format.
- **`DataPreprocessing.py`**: Handles all preprocessing tasks, including empty image removal, outlier detection, and scaling.
- **`unet_vgg16.py`**: Implements the U-Net model used for segmentation experiments.

## U-Net Segmentation Experiment
A U-Net model with VGG16 as the base was used to segment buildings from images before classification. However, this approach did not significantly improve the results.

## Results and Conclusion
Despite preprocessing and data augmentation improving the accuracy, extraneous elements in the images (e.g., cars, trees) introduced noise, limiting the model's ability to generalize. Future work could focus on better segmenting building areas to improve classification performance.

---

## Brief on How to Run the Codes in the Repository

### 1. Running `TrainAndaTestResnet50.py`
The script `traintestresnet50.py` is used to train and test a ResNet50 model on image data. It involves loading training and testing datasets, preprocessing them, and training the model with ResNet50 as the backbone. After training, predictions on test data are saved as a CSV file.

#### Steps to Run:
1. **Install Requirements**: Ensure the required libraries are installed by running:
    ```bash
    pip install -r requirements.txt
    ```

2. **Run the Script**: Use the command-line interface to provide paths to the training and testing directories:
    ```bash
    python traintestresnet50.py --train_dir path/to/train_data --test_dir path/to/test_data --output path/to/output.csv
    ```

#### Arguments:
- `--train_dir`: The directory containing the training data (in subfolders like A, B, C, D, S).
- `--test_dir`: The directory containing the testing data.
- `--output`: The path where the predictions will be saved (default is `predictions.csv`).

The script will:
- Load, preprocess, and normalize the images.
- Train a model using ResNet50 and augment the training data.
- Use callbacks like `ReduceLROnPlateau` and `EarlyStopping` to optimize training.
- Make predictions on the test dataset and save them as a CSV.

### 2. Running `DataPreprocessing.py`
The `DataPreprocessing.py` script performs preprocessing tasks and asks for user input during execution. It primarily involves outlier removal and image segmentation. The script will prompt for specific actions like whether to remove outliers and whether to generate image segments, and it also requires you to specify input and output directories.

#### Steps to Run:
1. **Install Requirements**: Make sure all required packages are installed using:
    ```bash
    pip install -r requirements.txt
    ```

2. **Run the Script**: Execute the script from the terminal:
    ```bash
    python DataPreprocessing.py
    ```

#### User Inputs During Execution:
Once the script starts, it will prompt you with the following questions:

- **Outlier Removal**: The script will ask if you want to remove outliers. You can choose "yes" or "no" depending on whether you'd like to remove any outliers in your dataset:
    ```bash
   Do you want to keep outliers in the dataset?(yes/no):
    ```

- **Segmentation Generation**: The script will ask if you want to generate segments using a U-Net model. Answer "yes" if you'd like to proceed with segment generation:
    ```bash
   Do you want to save segmentation masks? (yes/no):
    ```

- **Input and Output Directories**: You will be asked to provide paths to the directories where your input images are stored and where you want the output (preprocessed images or segmentation masks) to be saved:
    ```bash
    Enter the path for training data directory:
    Enter the path for cleaned data directory:
    ```

#### Example Execution:
- **Removing Outliers and Generating Segments**: If you want to remove outliers and generate segments, you would answer "no" and "yes" to questions respectively and provide the appropriate input and output directory paths when prompted.

- **Skipping Outlier Removal and Only Generating Segments**: If you're only interested in generating segments without removing outliers, you would answer "yes" to the first question and "yes" to the second question, then provide the necessary directories.

#### Sample Interaction:
```bash
python DataPreprocessing.py

Enter the path for training data directory:
> /path/to/input/images

Enter the path for cleaned data directory:
> /path/to/output

Do you want to keep outliers in the dataset?(yes/no):
> no

Do you want to save segmentation masks? (yes/no):
> yes


```
This will remove outliers from /path/to/input/images, perform segmentation, and save the processed images in /path/to/output.
