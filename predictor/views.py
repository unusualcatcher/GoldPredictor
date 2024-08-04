import os
import pickle
import numpy as np
import datetime
from django.conf import settings
from django.shortcuts import render

model_path = os.path.join(settings.BASE_DIR, 'model.pkl')
scaler_path = os.path.join(settings.BASE_DIR, 'scaler.pkl')

with open(model_path, 'rb') as f:
    model = pickle.load(f)
with open(scaler_path, 'rb') as s:
    scaler = pickle.load(s)


def getDate(Date):
    date_obj = datetime.datetime.strptime(Date, r"%Y-%m-%d")


    year = date_obj.year
    month = date_obj.month
    day = date_obj.day
    return year, month, day

def home(request):
    if request.method == "POST":
        d = request.POST.get('date')
        print(d)
        year, month, day = getDate(d)
        spx = float(request.POST.get('spx'))
        uso = float(request.POST.get('uso'))
        sil = float(request.POST.get('sil'))
        eur_usd = float(request.POST.get('eur-usd'))
        params = np.array([[spx, uso, sil, eur_usd]])
        parametres_transformed = scaler.transform(params)
        parametres = np.array([[year, month, day, 
                            parametres_transformed[0][0], 
                            parametres_transformed[0][1], 
                            parametres_transformed[0][2], 
                            parametres_transformed[0][3]]])
        prediction = model.predict(parametres)
        print(prediction)
        return render(request, 'predictor/index.html', context={'output':prediction[0]})

    return render(request, 'predictor/index.html')
