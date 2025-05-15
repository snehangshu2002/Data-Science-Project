# Fake News Classification App

This repository contains code for a Fake News Classification App that uses Machine Learning (ML), Deep Learning (Artificial Neural Network), and Streamlit to classify news as fake or true. The project is based on the [Medium article](https://medium.com/@snehangshubhuin2018/creating-a-fake-news-classification-app-using-machine-learning-deep-learning-streamlit-c893378d113b) by Snehangshu Bhuin.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Project Overview
The app classifies news articles as fake or true using ML and deep learning models. It includes data preprocessing, NLP with TF-IDF, model training, and a Streamlit web interface for user input and predictions.

## Features
- Classifies news as fake or true.
- Interactive Streamlit web app for real-time predictions.
- Includes EDA, NLP preprocessing, and model training pipelines.
- Modular code for easy modification.

## Dataset
The dataset is sourced from Hugging Face and contains labeled news articles. It is not included in this repository due to size constraints. Download it from [Hugging Face](https://huggingface.co/) and place it in the `data/` directory.

## Technologies Used
- Python 3.8+
- Scikit-learn (ML models, TF-IDF)
- TensorFlow/Keras (ANN)
- NLTK/SpaCy (NLP)
- Pandas/NumPy (data handling)
- Streamlit (web app)
- Matplotlib/Seaborn (EDA visualizations)

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/fake-news-classification.git
   cd fake-news-classification
   ```
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Download the dataset from Hugging Face and place it in `data/`.
5. (Optional) Train the model:
   ```bash
   python scripts/train_model.py
   ```
6. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Usage
1. Launch the app with `streamlit run app.py`. It opens at `http://localhost:8501`.
2. Enter news text in the text box.
3. Click "Classify" to see if the news is fake or true.

## Project Structure
```
fake-news-classification/
├── data/                    # Dataset files (download separately)
├── models/                  # Trained models
├── notebooks/               # EDA and experimentation notebooks
├── scripts/                 # Preprocessing and training scripts
│   ├── preprocess.py        # Data cleaning and NLP
│   ├── train_model.py       # Model training
├── app.py                   # Streamlit app
├── requirements.txt         # Dependencies
├── README.md                # This file
└── LICENSE                  # License file
```

## Contributing
1. Fork the repository.
2. Create a branch: `git checkout -b feature-branch`.
3. Commit changes: `git commit -m "Add feature"`.
4. Push: `git push origin feature-branch`.
5. Open a Pull Request.

## License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments
- Snehangshu Bhuin for the Medium article.
- Hugging Face for the dataset.
- Streamlit and open-source libraries used.
