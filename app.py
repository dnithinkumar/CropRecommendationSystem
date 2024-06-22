from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

app = Flask(__name__)



data = pd.read_csv('Crop_recommendation.csv')
#print(data)

features = ['N', 'P', 'K', 'rainfall','humidity','ph']




#X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33, random_state=42)

#rfc= RandomForestClassifier(criterion='gini', random_state=0)

#rfc.fit(X_train,y_train)
#pickle.dump(rfc,open('model.sav','wb'))

#print(y_pred)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Getting values from the form
    print(request.form['N'])
    N = float(request.form['N'])
    P = float(request.form['P'])
    K = float(request.form['K'])
    rainfall = float(request.form['rainfall'])
    humidity = float(request.form['humidity'])
    ph = float(request.form['pH'])
    
    
    input_data = pd.DataFrame([[N, P, K, rainfall,humidity,ph]],columns=features)
    
    
    rfc=pickle.load(open('model.sav','rb'))
    prediction = rfc.predict(input_data)
    print(prediction)

    return render_template('result.html',prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
