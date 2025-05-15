Fake News Classification App
This repository contains the code for a Fake News Classification App built using Machine Learning (ML), Deep Learning (Artificial Neural Network), and Streamlit for deployment. The app classifies news articles as fake or true based on user input. The project follows the methodology outlined in the Medium article by Snehangshu Bhuin.
Table of Contents

Project Overview
Features
Dataset
Technologies Used
Installation
Usage
Project Structure
Contributing
License
Acknowledgments

Project Overview
The Fake News Classification App is designed to combat misinformation by leveraging ML and deep learning techniques to classify news articles. The pipeline includes exploratory data analysis (EDA), data preprocessing, NLP with TF-IDF, model training (using Logistic Regression and ANN), and deployment as a web app using Streamlit. Users can input news text and receive a prediction on whether the news is fake or true.
Features

Text Classification: Classifies news as fake or true using trained ML and deep learning models.
Interactive Web App: Built with Streamlit for user-friendly input and real-time predictions.
Comprehensive Pipeline: Includes EDA, data cleaning, NLP preprocessing, and model training.
Modular Code: Organized scripts for preprocessing, model training, and app deployment.

Dataset
The dataset is sourced from Hugging Face and contains labeled news articles (fake or true). Due to size constraints, the dataset is not included in this repository. You can download it from the Hugging Face dataset hub or refer to the Medium article for the specific dataset link.
Technologies Used

Python 3.8+
Machine Learning: Scikit-learn (Logistic Regression, TF-IDF)
Deep Learning: TensorFlow/Keras (Artificial Neural Network)
NLP: NLTK or SpaCy (text preprocessing)
Data Handling: Pandas, NumPy
Web App: Streamlit
Visualization: Matplotlib, Seaborn (for EDA)
Environment Management: Pip, Virtualenv

Installation
Follow these steps to set up the project locally:

Clone the Repository:
git clone https://github.com/<your-username>/fake-news-classification.git
cd fake-news-classification


Create a Virtual Environment:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install Dependencies:
pip install -r requirements.txt


Download the Dataset:

Download the dataset from Hugging Face and place it in the data/ directory.
Update the dataset path in the preprocessing script if necessary.


Train the Model (Optional):

Run the training script to generate the model:python scripts/train_model.py




Run the Streamlit App:
streamlit run app.py



Usage

Launch the App:
After running streamlit run app.py, the app will open in your default browser at http://localhost:8501.


Input News Text:
Enter the news article text in the provided text box.


Get Prediction:
Click the "Classify" button to see whether the news is classified as fake or true.



Project Structure
fake-news-classification/
├── data/                    # Dataset files (not included, download from Hugging Face)
├── models/                  # Trained ML and ANN models
├── notebooks/               # Jupyter notebooks for EDA and experimentation
├── scripts/                 # Python scripts for preprocessing and training
│   ├── preprocess.py        # Data cleaning and NLP preprocessing
│   ├── train_model.py       # Model training (ML and ANN)
├── app.py                   # Streamlit app script
├── requirements.txt         # Project dependencies
├── README.md                # This file
└── LICENSE                  # License file

Contributing
Contributions are welcome! To contribute:

Fork the repository.
Create a new branch (git checkout -b feature-branch).
Make your changes and commit (git commit -m "Add feature").
Push to the branch (git push origin feature-branch).
Open a Pull Request.

Please ensure your code follows the project's coding standards and includes relevant documentation.
License
This project is licensed under the MIT License. See the LICENSE file for details.
Acknowledgments

Snehangshu Bhuin for the original Medium article.
Hugging Face for providing the dataset.
Streamlit for enabling rapid web app development.
The open-source community for the libraries used in this project.

