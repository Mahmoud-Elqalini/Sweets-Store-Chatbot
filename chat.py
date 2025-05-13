from flask import Flask, request, jsonify, render_template
import numpy as np
import torch
import torch.nn as nn
import re
import pyodbc
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from main_functions import (load_json_file, NeuralNet, bag_of_words, preprocess, evaluate_math_expression,
                            is_arabic, get_arabert_embedding, arabert_model)

app = Flask(__name__)

# Initialize Flan-T5
model_name = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Linear layer to transform embedding from 768 to 1152
embedding_transform = nn.Linear(768, 1176).to('cuda' if torch.cuda.is_available() else 'cpu')

# Connect to SQL Server
try:
    conn = pyodbc.connect(
        'DRIVER={SQL Server};'
        'SERVER=MAHMOUD-HAMDY;'
        'DATABASE=Sweets_Store;'
        'Trusted_Connection=yes;'
    )
    cursor = conn.cursor()
    print("Connected to SQL Server successfully")
except Exception as error:
    print(f"Error: Failed to connect to database: {str(error)}")
    exit()


# Database schema retrieval
def get_database_schema():
    schema_query = """
    SELECT TABLE_NAME, COLUMN_NAME 
    FROM INFORMATION_SCHEMA.COLUMNS 
    WHERE TABLE_CATALOG = 'Sweets_Store'
    """
    try:
        cursor.execute(schema_query)
        schema_info = cursor.fetchall()
        schema_dict = {}
        for table, column in schema_info:
            if table not in schema_dict:
                schema_dict[table] = []
            schema_dict[table].append(column)
        return schema_dict
    except Exception as e:
        print(f"Failed to retrieve schema: {str(e)}")
        return {}


# SQL query validation
def is_valid_sql_query(query):
    dangerous_keywords = ['DROP', 'DELETE', 'TRUNCATE', 'ALTER', 'INSERT', 'UPDATE']
    return not any(keyword in query.upper() for keyword in dangerous_keywords)


# Generate SQL query using Flan-T5
def generate_sql_with_dynamic_schema(user_request, language):
    schema_dict = get_database_schema()
    schema_text = "\n".join([f"- {table} ({', '.join(columns)})" for table, columns in schema_dict.items()])

    # Normalize Arabic text
    if language == 'ar':
        user_request = re.sub(r'[\u0617-\u061A\u064B-\u0652]', '', user_request)

    input_text = f"Convert this natural language query to SQL based on the schema: {schema_text}. Query: {user_request}"
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
    output = model.generate(**inputs, max_length=100, num_beams=5, no_repeat_ngram_size=2)
    sql_query = tokenizer.decode(output[0], skip_special_tokens=True)
    return sql_query


# Load intents
try:
    english_intents = load_json_file('file_training/english.json', 'English training data')
    arabic_intents = load_json_file('file_training/arabic.json', 'Arabic training data')
    print('Merge data successfully')
except Exception as error:
    print(f"Error: Failed to load intents: {str(error)}")
    exit()

# Load the trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Starting to load model...")
try:
    data = torch.load("best_model.pth", weights_only=False)
    model_state = data["model_state"]
    input_size = data["input_size"]
    hidden_size = data["hidden_size"]
    output_size = data["output_size"]
    all_words = data["all_words"]
    tags = data["tags"]

    # Move AraBERT model and embedding transform to the device
    arabert_model.to(device)
    embedding_transform.to(device)

    model = NeuralNet(input_size, hidden_size, output_size).to(device)
    model.load_state_dict(model_state)
    model.eval()
    print("Model loaded successfully!")
except FileNotFoundError:
    print("Error: best_model.pth not found! Please run train.py first.")
    exit()


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    if not user_input:
        return jsonify({"error": "No message provided"}), 400

    # Detect language
    language = 'ar' if is_arabic(user_input) else 'en'
    intents_to_use = arabic_intents if language == 'ar' else english_intents
    print(f"Input: {user_input}, Language: {language}")

    # Preprocess and predict
    if language == 'ar':
        # Get AraBERT embedding as PyTorch tensor
        try:
            embedding = get_arabert_embedding(user_input)
            # Transform and reshape to match input_size
            input_tensor = embedding_transform(embedding.unsqueeze(0)).to(device, dtype=torch.float32)
            # Verify embedding size matches model's expected input_size
            if input_tensor.size(1) != input_size:
                print(f"Embedding size {input_tensor.size(1)} does not match model input_size {input_size}")
                raise ValueError(f"Embedding size {input_tensor.size(1)} does not match model input_size {input_size}")
        except Exception as e:
            print(f"Arabic processing error: {str(e)}")
            response = f"آسف، فيه مشكلة في معالجة النص! الخطأ: {str(e)}" if language == 'ar' \
                else f"Sorry, there was an issue processing the text! Error: {str(e)}"
            return jsonify({"response": response})
    else:
        tokens = preprocess(user_input, language)
        bag_sentence = bag_of_words(tokens, all_words)
        input_tensor = torch.from_numpy(bag_sentence).reshape(1, -1).to(device, dtype=torch.float32)

    # Predict intent
    with torch.no_grad():
        output = model(input_tensor)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]
    print(f"Predicted tag: {tag}")

    # Handle math queries
    if tag == "math":
        try:
            result = evaluate_math_expression(user_input)
            response = f"الإجابة هي: {result}" if language == 'ar' else f"The answer is: {result}"
        except Exception as e:
            response = "آسف، مش قادر أحل المعادلة دي!" if language == 'ar' else "Sorry, I can't solve that equation!"
            print(f"Math evaluation error: {str(e)}")
        print(f"Response: {response}")
        return jsonify({"response": response})

    # Handle database queries
    if tag in ["product_availability", "offers_and_discounts", "product_price", "database_query"]:
        try:
            sql_query = generate_sql_with_dynamic_schema(user_request=user_input, language=language)
            print(f"Generated SQL Query: {sql_query}")
            if not is_valid_sql_query(sql_query):
                response = "آسف، الاستعلام مش آمن للتنفيذ." if language == 'ar' \
                    else "Sorry, the generated query is not safe to execute."
                print("Unsafe SQL query detected")
                return jsonify({"response": response})

            cursor.execute(sql_query)
            result = cursor.fetchall()
            if result:
                if tag == "product_availability":
                    stock = result[0][0]
                    product_name = user_input.split()[-2] if language == 'ar' else user_input.lower().split()[-2]
                    if stock > 0:
                        response = f"نعم، عندنا {product_name} في المخزون!" if language == 'ar' \
                            else f"Yes, we have {product_name.capitalize()} in stock!"
                    else:
                        response = f"آسف، {product_name} مش متوفر في المخزون." if language == 'ar' \
                            else f"Sorry, {product_name.capitalize()} is out of stock."
                elif tag == "offers_and_discounts":
                    discount = result[0][0]
                    response = f"فيه عرض بـ {discount}% دلوقتي!" if language == 'ar' \
                        else f"There’s a {discount}% discount available now!"
                elif tag == "product_price":
                    price = result[0][0]
                    product_name = user_input.split()[-2] if language == 'ar' else user_input.lower().split()[-2]
                    response = f"سعر {product_name} هو {price} جنيه." if language == 'ar' \
                        else f"The price of {product_name.capitalize()} is ${price}."
                else:
                    response = str(result)  # Generic response for other database queries
            else:
                response = "آسف، ما لقيتش نتايج للطلب ده." if language == 'ar' \
                    else "Sorry, I couldn't find any results for your request."
            print(f"Response: {response}")
            return jsonify({"response": response})
        except Exception as e:
            response = f"خطأ في تنفيذ الاستعلام: {str(e)}" if language == 'ar' else f"Error executing query: {str(e)}"
            print(f"Database query error: {str(e)}")
            return jsonify({"response": response})

    # Handle general intents
    for intent in intents_to_use['intents']:
        if intent['tag'] == tag:
            response = np.random.choice(intent['responses'])
            print(f"Response: {response}")
            return jsonify({"response": response})

    # Fallback response
    response = "مش فاهم، ممكن تعيد صياغة السؤال؟" if language == 'ar' else "I don't understand. Can you rephrase?"
    print(f"Fallback response triggered for input: {user_input}")
    return jsonify({"response": response})


if __name__ == ('__'
                'main__'):
    app.run(debug=True, use_reloader=False)
