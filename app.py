from flask import Flask, request, render_template, redirect
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

data = pd.read_csv('files/data.csv')
X = data['uses']
y = data['Generic Name']
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

def medicine(s):
    user_input_vectorized = vectorizer.transform([s])
    similarity_scores = cosine_similarity(user_input_vectorized, X)
    most_similar_index = similarity_scores.argmax()
    if most_similar_index > 0:
        recommended_medication = data.loc[most_similar_index, 'Generic Name']
        return recommended_medication
    else:
        return 'Sorry, we cannot find a medicine for your condition'


@app.route('/')
def index():
    return redirect('/process')

@app.route('/process', methods=['GET', 'POST'])
def process():
    if request.method == 'POST':
        user_input = request.form['input_data']
        s = user_input.split()
        fi = ''
        for i in s:
            fi = fi + i

        result = medicine(fi)
        return render_template('result.html', results=result)
    
    # Handle GET request
    return render_template('sample.html', results=None)






if __name__ == '__main__':
    app.run(debug=True)