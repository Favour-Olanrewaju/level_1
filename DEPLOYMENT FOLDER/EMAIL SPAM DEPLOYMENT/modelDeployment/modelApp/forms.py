from django import forms

class ModelForm(forms.Form):
    data = forms.FileField(required=False)
    pclass = forms.CharField(max_length=10, label="Passanger class", required=False)
    sex = forms.CharField(max_length=10, label="Sex", required=False)
    age = forms.CharField(max_length=10, label="Age", required=False)
    sibsp = forms.CharField(max_length=10, label="Sibling", required=False)
    parch = forms.CharField(max_length=10, label="Alone", required=False)
    fare = forms.CharField(max_length=10, label="Tfare", required=False)
    embacked = forms.CharField(max_length=10, label="Location", required=False)
    