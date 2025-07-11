from django.urls import path
from django.contrib.auth import views as auth_views
from . import views

app_name = 'accounts'

urlpatterns = [
    path('', views.dashboard, name='dashboard'),
    path('register/', views.register, name='register'),
    path('login/', auth_views.LoginView.as_view(template_name='accounts/login.html'), name='login'),
 path('logout/', views.custom_logout, name='logout'),  # Use custom logout
    path('api/training-progress/', views.api_training_progress, name='api_training_progress'),
]