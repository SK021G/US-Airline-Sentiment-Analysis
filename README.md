# Airline Tweet Sentiment Analysis with RNN ✈️

This project uses a Recurrent Neural Network (RNN) to classify tweets about US airlines into three categories: **positive**, **negative**, or **neutral**. The model is built with TensorFlow/Keras and trained on a public dataset of airline tweets.



## 📝 Overview

The goal of this project is to analyze the sentiment of customer feedback on Twitter. The provided Jupyter Notebook, `Airline_Sentiment_Analysis.ipynb`, walks through the entire process from data loading and cleaning to model training and real-time prediction.

---

## ✨ Features

* **Data Cleaning**: Raw tweet text is cleaned by removing @mentions, URLs, and special characters.
* **Text Preprocessing**: Cleaned text is tokenized and padded to create uniform numerical sequences suitable for the neural network.
* **RNN Model**: A simple RNN model is built using a Keras `Sequential` model, including an `Embedding` layer, a `SimpleRNN` layer, and a `Dense` output layer.
* **Model Training**: The model is trained on 80% of the dataset and validated on the remaining 20%.
* **Real-Time Prediction**: A function is included that allows you to input any tweet and get an instant sentiment prediction with a confidence score.

---

## 🛠️ Technologies Used

* **Python 3**
* **TensorFlow & Keras**: For building and training the neural network.
* **Scikit-learn**: For splitting the data.
* **Pandas**: For data manipulation and loading the CSV.
* **NumPy**: For numerical operations.
* **Jupyter Notebook**: For interactive development.

---

## 🧠 Model Architecture

The neural network consists of three main layers:

1.  **Embedding Layer**: Converts integer-encoded words into dense vectors of a fixed size (`output_dim=32`). It's the first hidden layer that learns a meaningful representation for each word in the vocabulary.
2.  **SimpleRNN Layer**: Processes the sequence of word vectors, capturing temporal patterns and context from the tweet (`units=32`).
3.  **Dense Output Layer**: A fully connected layer with 3 neurons (one for each sentiment class) and a `softmax` activation function to output a probability distribution across the classes.

---

## 🚀 How to Run

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/twitter-sentiment-analysis.git](https://github.com/your-username/twitter-sentiment-analysis.git)
    cd twitter-sentiment-analysis
    ```

2.  **Install the required libraries:**
    ```bash
    pip install pandas numpy scikit-learn tensorflow
    ```

3.  **Download the Dataset:**
    This project uses the "Tweets.csv" file from the [U.S. Airline Sentiment](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment) dataset on Kaggle. **You must place the `Tweets.csv` file in the root directory of this repository.**

4.  **Run the Jupyter Notebook:**
    Launch the notebook and run the cells in order.
    ```bash
    jupyter notebook Airline_Sentiment_Analysis.ipynb
    ```
    The last cell provides an interactive prompt to test your own tweets!
