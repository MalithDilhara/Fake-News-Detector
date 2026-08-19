from flask import Flask, render_template, request
import pickle
import re
import string

app = Flask(__name__)

# Save කරපු Models ලෝඩ් කිරීම
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Text Preprocessing function එක
def wordopt(text):
    text = text.lower()
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"\W", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>+", "", text)
    text = re.sub(r"[%s]" % re.escape(string.punctuation), "", text)
    text = re.sub(r"\n", "", text)
    text = re.sub(r"\w*\d\w*", "", text)
    return text

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        news = request.form['news']
        if not news.strip():
            return render_template('index.html', prediction_text="කරුණාකර ප්‍රවෘත්තියක් ඇතුළත් කරන්න.")
            
        # ප්‍රවෘත්තිය පිරිසිදු කර පරීක්ෂා කිරීම
        testing_news = wordopt(news)
        new_xv_test = vectorizer.transform([testing_news])
        prediction = model.predict(new_xv_test)[0]
        
        if prediction == 0:
            result = "❌ FAKE NEWS (අසත්‍ය ප්‍රවෘත්තියක්)"
            color = "red"
        else:
            result = "✅ REAL NEWS (සත්‍ය ප්‍රවෘත්තියක්)"
            color = "green"
            
        return render_template('index.html', prediction_text=result, result_color=color, original_text=news)

if __name__ == "__main__":
    app.run(port=5001)
    