from flask import Flask, render_template, request
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import re

app = Flask(__name__, template_folder='templates')

model =  tf.keras.models.load_model('next_word_prediction.h5')
with open("tokenizer.pkl", 'rb') as file:
    tokenizer = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')




@app.route('/generate', methods=['GET', 'POST'])
def generate():

    # If a form is submitted
    if request.method == "POST":
        
        # Get values through input bars
        text = request.form.get("Text")
        predict_next_words = request.form.get("NoOfWords")
        predict_next_words =int(predict_next_words)
        # Get prediction from the generate_text function written above
       
        for _ in range(predict_next_words):
            token_list = tokenizer.texts_to_sequences([text])[0] 
            token_list = pad_sequences([token_list],maxlen= 291, padding = 'pre')
            predicted = np.argmax(model.predict(token_list,verbose=0), axis = 1)
            output_word = ""
            for word, index in tokenizer.word_index.items(): 
                if index== predicted:
                    output_word = word
                    break
            text += " " + output_word
        
        
            
        return render_template("generate.html", output = text)


# Running the app
if __name__ == "__main__":
    app.run(debug=True, port=4700)
