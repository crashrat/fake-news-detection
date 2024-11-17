from flask import Flask, render_template, request
import pickle
import os

app = Flask(__name__)

# Define the paths for the vectorizer and model
vectorizer_path = 'C:\\Users\\mozoo\\OneDrive\\Desktop\\ahmed project,s\\Fake_News_Detection\\tfidf_vectorizer.pkl'
model_path = 'C:\\Users\\mozoo\\OneDrive\\Desktop\\ahmed project,s\\Fake_News_Detection\\model.pkl'

# Load vectorizer and model with error handling
if os.path.exists(vectorizer_path):
    with open(vectorizer_path, 'rb') as vectorizer_file:
        tfvect = pickle.load(vectorizer_file)
else:
    print(f"Error: File not found: {vectorizer_path}")
    tfvect = None  # Set to None or handle appropriately

if os.path.exists(model_path):
    with open(model_path, 'rb') as model_file:
        loaded_model = pickle.load(model_file)
else:
    print(f"Error: File not found: {model_path}")
    loaded_model = None  # Set to None or handle appropriately

def fake_news_det(news):
    if tfvect is None or loaded_model is None:
        return "Model not loaded correctly."
    
    vectorized_input_data = tfvect.transform([news])
    prediction = loaded_model.predict(vectorized_input_data)
    return prediction

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        pred = fake_news_det(message)
        return render_template('index.html', prediction=pred)
    else:
        return render_template('index.html', prediction="Something went wrong")

if __name__ == '__main__':
    app.run(debug=True)