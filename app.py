
from flask import Flask,request,jsonify
from flask.templating import render_template
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
nltk.download('punkt')

import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)



ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

# st.title("Email/SMS Spam Classifier")

# input_sms = st.text_area("Enter the message")

# if st.button('Predict'):

#     # 1. preprocess
#     transformed_sms = transform_text(input_sms)
#     # 2. vectorize
#     vector_input = tfidf.transform([transformed_sms])
#     # 3. predict
#     result = model.predict(vector_input)[0]
#     # 4. Display
#     if result == 1:
#         st.header("Spam")
#     else:
#         st.header("Not Spam")

@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict', methods=["GET","POST"])
def get_click_prediction():
    if request.method == 'POST':

        try:
            text = request.form.get('message')

            # 1. preprocess
            transformed_sms = transform_text(text)
            # 2. vectorize
            vector_input = tfidf.transform([transformed_sms])
            # 3. predict
            result = model.predict(vector_input)[0]
            # 4. Display
            if result == 1:
                # st.header("Spam")
                pred = 'spam'
            else:
                # st.header("Not Spam")
                pred = 'Ham'

            return render_template('predict.html',predictedd = "The Final Flag is : {}".format(pred))

        except Exception as e:
            return('Something is not right!:'+str(e))
        
    else:
         return render_template('predict.html')
    
@app.route("/result",methods=['GET', 'POST'])
def last():
    return render_template('final.html')


if(__name__ == '__main__'):
    app.run()


