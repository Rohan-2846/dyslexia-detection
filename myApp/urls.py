from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static


urlpatterns = [
    path('', views.index, name='index'),
    path('about', views.about, name='about'),
    path('contact', views.contact, name='contact'),
    path('query', views.query, name='query'),
    path('sessions', views.sessions, name='sessions'),

    path('logout1/', views.logout1, name='logout1'),
    path('register/', views.register, name='register'),
    path('login_view/', views.login_view, name='login_view'),

    path('detect', views.detect, name='detect'),
    path('predict_dyslexia', views.predict_dyslexia, name='predict_dyslexia'),
]

urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
