# 🍬 Sweets Store Chatbot

## 🧠 Overview
The **Sweets Store Chatbot** is an AI-powered conversational assistant for a sweets store. It supports both **English and Arabic**, allowing users to ask about prices, availability, discounts, order tracking, and more. The chatbot uses NLP and machine learning to classify intents, generate SQL queries, and interact with a SQL Server database. It also includes a math-solving feature for basic arithmetic.

This project integrates advanced NLP models such as **AraBERT** for Arabic and **Flan-T5** for SQL query generation, alongside a custom-trained neural network for intent classification.

---

## ✨ Features

- 🌐 **Bilingual Support**: Handles queries in both English and Arabic.
- 🎯 **Intent Classification**: Trained neural network (94.16% accuracy) detects user intents like `order_tracking`, `product_price`, etc.
- 🗄️ **Database Integration**: Connects to a SQL Server (`Sweets_Store`) to retrieve real-time data.
- 🔍 **SQL Query Generation**: Flan-T5 translates natural language into SQL queries.
- ➗ **Math Solver**: Solves simple arithmetic (e.g., `2 + 3`, `٢ زائد ٣`).
- 🔐 **Secure Query Handling**: Prevents unsafe operations like `DROP`, `DELETE`.

---

## 🗂️ Project Structure

- `chat.py` – Flask app for chatbot interface.
- `main_functions.py` – Core functions: preprocessing, embeddings, math solving, model loading.
- `train.py` – Trains the intent classification model.
- `classification_report.txt` – Model performance report.
- `confusion_matrix.png` for detailed performance.
- `file_training/` – Contains `english.json` and `arabic.json` (training data).

---

## ⚙️ Installation

### 🔧 Prerequisites

- Python 3.8+
- SQL Server
- CUDA-enabled GPU (optional)

### 🚀 Steps

1. **Clone the Repository**
    ```bash
    git clone https://github.com/Mahmoud-Elqalini/sweets-store-chatbot.git
    cd sweets-store-chatbot
    ```

2. **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```
    > 💡 You may need to install `pyodbc` separately depending on your OS.

3. **Configure SQL Server**
    - Update connection details in `chat.py`.
    - Ensure a `Sweets_Store` database exists with relevant tables.

4. **Train the Intent Classifier**
    ```bash
    python train.py
    ```

5. **Run the Chatbot**
    ```bash
    python chat.py
    ```
    - App runs on `http://localhost:5000`.

---

## 💬 Usage

- **Web Interface**: Access via browser (`index.html`).
- **Example Queries**:
  - English: `What is the price of the cake?`
  - Arabic: `سعر الكيكة كام؟`
  - Math: `What is 5 times 3?` or `٥ مضروب في ٣ كام؟`
  - and more 
---

## 📊 Model Performance

- **Accuracy**: 94.16%
- **Macro F1-Score**: 0.9397
- **Weighted F1-Score**: 0.9415  
> 📄 See `classification_report.txt` for detailed performance.
> 📄 See `confusion_matrix.png` for detailed performance.

---

## 🛠 Technologies Used

- **Backend**: Python (Flask, PyTorch)
- **NLP**: AraBERT, Flan-T5 (Hugging Face)
- **Libraries**: NLTK, SpaCy, Stanza, SymPy, NLPAug
- **Database**: SQL Server
- **Frontend**: Basic HTML (Flask template)

---

## ⚠️ Limitations

- Requires SQL Server connection.
- Only recognizes intents defined in training files.
- Math solver supports only simple arithmetic.
- Arabic parsing may struggle with complex structures.

---

## 🔮 Future Improvements

- Add support for more languages.
- Enhance math solver for advanced equations.
- Improve frontend using React or JavaScript.
- Expand database and intent variety.

---

## 🤝 Contributing

Contributions are welcome!  
Fork the repo, create a new branch, and submit a pull request.

---

## 📬 Contact

For questions or collaboration:

- LinkedIn: (https://www.linkedin.com/in/mahmoud-elqalini-012749286)
- Email: `mahmoudeq02@gmail.com`

---

> Made with ❤️ by Mahmoud Elqalini
