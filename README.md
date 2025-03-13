# Chatbot using NLP and Logistic Regression

## Overview
This project implements a chatbot using **Natural Language Processing (NLP)** and **Logistic Regression**. It uses **TF-IDF Vectorization** to process text and train a classifier for intent recognition. The chatbot is built with **Streamlit**, a Python framework for interactive web applications.

## Features
- **Machine Learning-Based Intent Recognition**: Uses TF-IDF and Logistic Regression.
- **Interactive Chat Interface**: Built using **Streamlit**.
- **Conversation Logging**: Stores chat history in a CSV file.
- **Pretrained Model**: Improves accuracy using **ngrams** and **stop word filtering**.
- **Graceful Exit**: Detects goodbye messages and ends the session politely.

## Technologies Used
- **Python**
- **Natural Language Toolkit (NLTK)**
- **Scikit-learn**
- **Streamlit**
- **JSON for storing intents**
- **CSV for logging conversations**

## Setup Instructions
### 1. Clone the Repository
```bash
git clone https://github.com/Chefjdeep/AICTE_NLP_ChatBot
cd AICTE_NLP_ChatBot
```

### Create a Virtual Environment (Optional but Recommended)
```bash
python -m venv chatbot_env
source chatbot_env/bin/activate  # On Windows: chatbot_env\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Chatbot
```bash
streamlit run chatbot.py
```

## Project Structure
```
├── AICTE_NLP_ChatBot.ipynb    # Jupyter Notebook for development and testing
├── LICENSE                    # Project License
├── README.md                  # Project Documentation
├── chat_log.csv               # Stores conversation history
├── chatbot.py                 # Main chatbot implementation
├── feedback_log.csv           # Stores user feedback on chatbot responses
├── intents.json               # Predefined intents and responses
├── requirements.txt           # Required dependencies
```

## 📊 Usage

1. Open the Streamlit UI and enter your query in the chat interface.
2. The chatbot will analyze the intent and provide an response.
3. Conversations are automatically logged in `chat_log.csv`.
4. Users can provide feedback on chatbot responses.
5. Conversation history can be accessed via the "Conversation History" menu.

## 📂 Dataset: `intents.json`

The chatbot uses a JSON-based dataset where:
- `patterns` contain sample user inputs.
- `responses` contain possible chatbot replies.
- `tag` is used for classification.

## Future Improvements to be made
- Integrate **Deep Learning (LSTMs or Transformers)** for better intent classification.
- Implement **database storage** for chat history.
- Add **speech-to-text** for voice interactions.
- Deploy chatbot using **Render or Heroku**.

## License
This project is licensed under the **MIT License**.

*Developed as part of the AICTE Internship Project.*