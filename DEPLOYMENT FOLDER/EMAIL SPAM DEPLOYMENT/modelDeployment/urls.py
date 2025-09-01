
from django.contrib import admin
from django.urls import path
from modelDeployment.modelApp import views as mod

urlpatterns = [
    path("admin/", admin.site.urls),
    path("", mod.predict, name="model"),
]
