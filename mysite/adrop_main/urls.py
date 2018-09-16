from django.urls import path
from . import views
from django.conf.urls import url

urlpatterns = [
    path('',views.index, name = 'index'),
    path('logon/',views.logonValidation, name='logonValidation'),
    path('uploadImage/',views.uploadImage, name='uploadImage'),
    path('compareAndSave/',views.compareAndSave,name="compareAndSave"),
    path('compare/',views.compare,name="compare"),
    path('showImage/',views.showImage, name='showImage'),

]