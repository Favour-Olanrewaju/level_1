from django.shortcuts import render
from .forms import ModelForm
from pickle_blosc import unpickle
import pandas as pd
from sklearn.preprocessing import StandardScaler


# Create your views here.
# def predict(request):
#     if request.method == "POST":
#         form = ModelForm(request.FILES or None)   
#         model = pickle.load(open("C://pythonclass//djangoclass//modelDeployment//modelDeployment//modelApp//titanic_model.sav", "rb"))
#         data = request.FILES['data']
#         survival = pd.read_csv(data)
#         pred = model.predict(survival)
#         if pred == 0:
#             output = "Oh no! You didn't make it"
#         else:
#             form = ModelForm()
#             output = "Nice! You survived"
#         return render(request, 'model.html', {"pred":output, 'form':form})
#     else:
#         form = ModelForm()
#         return render(request, 'model.html', {'form': form})


def predict(request):
    if request.method == "POST":
        form = ModelForm(request.POST or None) 
        
        
        model = unpickle("C://pythonclass//djangoclass//modelDeployment//modelDeployment//modelApp//titanic_model.pkl")
        sc = unpickle("C://pythonclass//djangoclass//modelDeployment//modelDeployment//modelApp//scale_model.pkl")
        
        if form.is_valid():
            pclass = form.cleaned_data['pclass']
            sex = form.cleaned_data['sex']
            age = form.cleaned_data['age']
            sibsp = form.cleaned_data['sibsp']
            parch = form.cleaned_data['parch']
            fare = form.cleaned_data['fare']
            embacked = form.cleaned_data['embacked']

            survival = [[float(pclass), float(sex), float(age), float(sibsp), float(parch), float(fare), float(embacked)]]
        
        survival = sc.transform(survival)

        pred = model.predict(survival)
        if pred == 0:
            form = ModelForm()
            output = "Oh no! You didn't make it"
        else:
            form = ModelForm()
            output = "Nice! You survived"
        return render(request, 'model.html', {"pred":output, 'form':form})
    else:
        form = ModelForm()
        return render(request, 'model.html', {'form': form})
