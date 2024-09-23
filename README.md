# ASL Gesture Recognizer

This project implements a gesture recognition system for American Sign Language (ASL) using the **Inception-ResNet V2** architecture in TensorFlow. The model is trained to recognize various ASL gestures from image inputs. Additionally, a graphical user interface (GUI) is included, which allows users to upload images for recognition, with output provided both as text and speech.

## Requirements

Ensure the following dependencies are installed:

- **TensorFlow (version 2.0+)**
- **OpenCV**
- **Numpy**
- **gTTS** (Google Text-to-Speech for GUI)
- **Tkinter** (For GUI)

## Instructions

### 1. Clone the Repository
Start by cloning the repository to your local machine.

### 2. Download the Dataset
Download the ASL gesture dataset from [Kaggle](https://www.kaggle.com/datasets/ayuraj/asl-dataset).

### 3. Train the Model
To train the model, use the following command:

```bash
python main.py \
--train_path ./path_to_data_directory \
--mode train \
--epochs 20
```

Replace `./path_to_data_directory` with the actual path to your dataset.

### 4. Test the Model
After training, you can test the model using:

```bash
python main.py \
--train_path ./path_to_data_directory \
--mode test
```

### 5. Run the GUI
This project includes a GUI for easier interaction with the model. You can upload images through the interface, and the recognized gesture will be displayed as text and also relayed via speech using the gTTS module. 

**Note:** An active internet connection is required for the speech output, as the gTTS module relies on online resources.

To start the GUI, run the following script:

```bash
python guiofasl.py
```

## Conclusion

The ASL Gesture Recognizer utilizes a powerful deep learning model to identify American Sign Language gestures accurately. The inclusion of a GUI with both text and speech output makes it user-friendly and accessible, enhancing the user experience.