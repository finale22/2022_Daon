from django.urls import path, include

from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('main/', views.main, name='main'),
    path('recomand/', views.recomand, name='recomand'),
    path('make/', views.make, name='make')

]