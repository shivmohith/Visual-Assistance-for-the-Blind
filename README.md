# Visual Assistance for the Blind
This project was carried out in the months of Jan-May 2020.

## Description
The estimated number of visually impaired people in the world is 285 million in which 39 million people are blind and 246 million people have low vision. This project aims at assisting visually impaired people through DL by providing a system which can describe the surrounding as well as answer questions about the surrounding of the user.
It comprises two models, a Visual Question Answering (VQA) model and an image captioning model.
The image captioning model takes in an image as the input and generates the caption describing the image.
The VQA model is fed with an image and a question and it predicts the answer to the question asked with respect to the image.

### Dataset
For Image captioning model, the dataset used is the [MS-COCO](http://cocodataset.org/#download) dataset.
For VQA model, the datasets used are [MS-COCO](http://cocodataset.org/#download) dataset and [VQA](https://visualqa.org/vqa_v1_download.html) dataset.

## Methodology
1. Build and train Image captioning model
2. Build and train VQA model
3. Construct Speech to Text and Text to Speech models
4. Integrate all the models to form a single product

In real time, the user will be provided with an option to choose between describing the surroundings or ask a question about the
surroundings and he/she will be provided with an answer.

## Running the application

### Software Requirements
- Python version 3 with necessary libraries as mentioned in requirements.txt files
- Cuda tooklkit version 10.2 if you need to train on GPU, update driver

### Hardware Requirements
- PC with a good NVidia GPU and webcam

### Folder Strcuture
- Image_Captioning directory - Contains the files for training and testing the image captioning model. This directory is also needed while running the real time code.
- VQA - Contains 2 directories for traning and testing MLP and CNN_LSTM based VQA models. Under each directory you will find main.py which contains both the training and testing code.
**Note: It is advised to run the VQA codes in Google Colab to avoid unnecessary errors.**
- There are 3 python files inside the root directory which is for real time demo.
    - [conversions.py](https://github.com/Shivmohith/Visual-Assistance-for-the-Blind/blob/master/conversions.py): Contains the code for Speech to Text and Text to Speech components
    - [models.py](https://github.com/Shivmohith/Visual-Assistance-for-the-Blind/blob/master/model.py): Contains the Image captioning and VQA models
    - [product.py](https://github.com/Shivmohith/Visual-Assistance-for-the-Blind/blob/master/product.py): Contains the real time code using webcam


### To train and test the image captioning do the following steps
1. Download the train and test dataset (images and annotations 2014) from "http://cocodataset.org/#download"
2. Move into Source Code/Image_Captioning directory
3. Install the requirements using the command "pip install -r requirements.txt"
4. Replace all the required file paths to your convenience
5. Uncomment line no. 79 & 80 in train.py and run it to generate the image batch features and train the model. To run the code again, comment the lines to avoid generating the batch features again.
6. Run test.py to test the model 

### To train and test the MLP VQA model do the following steps
1. Move into VQA/MLP directory
2. Download the train and validation MS-COCO images (2014) from "http://cocodataset.org/#download"
3. Download the train and validation Questions (2015) and Answers (2015) from "https://visualqa.org/vqa_v1_download.html"
4. Download the image features pickle file for train images from "https://drive.google.com/file/d/1icMniCVK8D3pGoDgkBkTl7K2zTsXRf13/view?usp=sharing"
5. Download the image features pickle file for validation from "https://drive.google.com/file/d/1sa_ZEej11NFtiAnmhR18X5o6_Ctc6qcI/view?usp=sharing"
6. Download the preprocessed dataset from "https://drive.google.com/drive/folders/1LmOr3poPLLBLDF0e3z50XeMHKmnsQzqI?usp=sharing"
7. Install the requirements using the command "pip install -r requirements.txt"
8. Run the main.py to train and test

### To train and test the CNN_LSTM VQA model do the following steps
1. Move into Source Code/VQA/CNN_LSTM directory
2. Download the train and validation MS-COCO images (2014) from "http://cocodataset.org/#download"
3. Download the train and validation Questions (2015) and Answers (2015) from "https://visualqa.org/vqa_v1_download.html"
4. Download the image features pickle file for train images from "https://drive.google.com/file/d/1icMniCVK8D3pGoDgkBkTl7K2zTsXRf13/view?usp=sharing"
5. Download the image features pickle file for validation from "https://drive.google.com/file/d/1sa_ZEej11NFtiAnmhR18X5o6_Ctc6qcI/view?usp=sharing"
6. Download the preprocessed dataset from "https://drive.google.com/drive/folders/1LmOr3poPLLBLDF0e3z50XeMHKmnsQzqI?usp=sharing"
7. Install the requirements using the command "pip install -r requirements.txt"
8. Run the main.py to train and test

### To test the code in real time with webcam
1. Install the requirements using the command "pip install -r requirements.txt" present in the root directory
2. Run [product.py](https://github.com/Shivmohith/Visual-Assistance-for-the-Blind/blob/master/product.py) code
 
