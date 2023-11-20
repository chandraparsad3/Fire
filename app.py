import pickle
from flask import Flask,render_template,request,jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
app = Flask(__name__)


ridge=pickle.load(open('models/ridge.pkl','rb'))
scaler=pickle.load(open('models/scaler.pkl','rb'))

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/predictdata',methods=["POST","GET"])
def predict_datapoint():
    if request.method=="POST":
        temperature = float(request.form.get('temperature'))
        rh = float(request.form.get('rh'))
        ws = float(request.form.get('ws'))
        rain = float(request.form.get('rain'))
        ffmc = float(request.form.get('ffmc'))
        dmc = float(request.form.get('dmc'))
        isi = float(request.form.get('isi'))
        classes=float(request.form.get('classes'))
        region = float(request.form.get('region'))
       
        new_data=scaler.transform([[temperature,rh,ws,rain,ffmc,dmc,isi,classes,region]])
        result=ridge.predict(new_data)

        return render_template('home.html',result=result[0])
    else:
        return render_template('home.html')

if __name__=="__main__":
    app.run(host="0.0.0.0")
