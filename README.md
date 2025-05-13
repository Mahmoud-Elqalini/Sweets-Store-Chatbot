Sweets Store Chatbot
Overview
The Sweets Store Chatbot is an AI-powered conversational agent designed to assist customers of a sweets store. It supports both English and Arabic languages, enabling users to interact naturally by asking about product prices, availability, discounts, order tracking, and more. The chatbot leverages natural language processing (NLP) and machine learning to classify user intents, process queries, and fetch data from a SQL Server database. It also includes a math-solving feature for simple arithmetic queries.
This project demonstrates the integration of advanced NLP models like AraBERT for Arabic text processing and Flan-T5 for generating SQL queries from natural language, alongside a custom-trained neural network for intent classification.
Features

Bilingual Support: Handles user queries in both English and Arabic.
Intent Classification: Identifies user intents (e.g., order_tracking, product_price) with a trained neural network (accuracy: 94.16%).
Database Integration: Connects to a SQL Server database (Sweets_Store) to fetch real-time data like product prices and availability.
SQL Query Generation: Uses Flan-T5 to convert natural language queries into SQL queries for database access.
Math Solver: Evaluates simple arithmetic expressions (e.g., "2 + 3" or "٢ زائد ٣").
Secure Query Handling: Validates SQL queries to prevent unsafe operations (e.g., DROP, DELETE).

Project Structure

chat.py: Main Flask application for running the chatbot and handling user requests.
main_functions.py: Contains core functions for text preprocessing, AraBERT embeddings, math evaluation, and neural network definition.
train.py: Script for training the intent classification model, including data augmentation and performance evaluation.
classification_report.txt: Output of the model's performance metrics on the validation set.
file_training/: Directory containing english.json and arabic.json for training data (intents, patterns, responses).

Installation
Prerequisites

Python 3.8+
SQL Server (for database integration)
CUDA-enabled GPU (optional, for faster training/inference)

Steps

Clone the Repository
git clone https://github.com/your-username/sweets-store-chatbot.git
cd sweets-store-chatbot


Install Dependencies
pip install -r requirements.txt

Note: You may need to install pyodbc drivers for SQL Server separately depending on your OS.

Set Up the Database

Configure the SQL Server connection in chat.py (update SERVER, DATABASE, etc.).
Ensure the Sweets_Store database exists with relevant tables (e.g., products, prices).


Train the Model
python train.py

This generates best_model.pth and data.pth for the trained model.

Run the Chatbot
python chat.py

The Flask app will run on http://localhost:5000. Open the browser to interact with the chatbot via the provided index.html.


Usage

Web Interface: Access the chatbot through the Flask app's web interface (index.html).
Example Queries:
English: "What is the price of the cake?"
Arabic: "سعر الكيكة كام؟"
Math: "What is 5 times 3?" or "٥ مضروب في ٣ كام؟"



The chatbot will classify the intent, fetch data if needed (e.g., price from the database), or solve the math query, and respond accordingly.
Model Performance
The intent classification model was trained on augmented English and Arabic datasets. Key metrics from the validation set:

Accuracy: 94.16%
Macro Avg F1-Score: 0.9397
Weighted Avg F1-Score: 0.9415

For detailed performance per intent, refer to classification_report.txt.
Technologies Used

Python Libraries: Flask, PyTorch, Transformers (Hugging Face), NumPy, NLTK, SpaCy, Stanza, SymPy, NLPAug
NLP Models: AraBERT (for Arabic embeddings), Flan-T5 (for SQL query generation)
Database: SQL Server
Frontend: Basic HTML (Flask template)

Limitations

Requires a SQL Server database to function fully.
Limited to intents defined in english.json and arabic.json.
Math solver supports basic arithmetic only.
Arabic query handling may fail for complex sentences due to AraBERT's tokenization limits.

Future Improvements

Add support for more languages.
Enhance the math solver for advanced equations.
Implement a more robust frontend with JavaScript/React.
Expand the database schema for more complex queries.

Contributing
Contributions are welcome! Please fork the repository, create a new branch, and submit a pull request with your changes.
License
This project is licensed under the MIT License. See the LICENSE file for details.
Contact
For inquiries, reach out via LinkedIn or email at your-email@example.com.
