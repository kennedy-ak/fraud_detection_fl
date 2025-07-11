# from django.urls import path
# from . import views

# app_name = 'models'

# urlpatterns = [
#     path('upload/', views.upload_dataset, name='upload_dataset'),
#     path('training/<str:session_id>/', views.training_progress, name='training_progress'),
#     path('api/training-status/<str:session_id>/', views.api_training_status, name='api_training_status'),
#     path('debug/active-sessions/', views.debug_active_sessions, name='debug_active_sessions'),

#      # Model download endpoints
#     path('download/<str:session_id>/<str:model_type>/', views.download_model, name='download_model'),
#     path('api/global-model-status/', views.check_global_model_availability, name='global_model_status'),
#     # Admin endpoints
#     path('admin/download-global/<int:round_id>/', views.download_global_model_admin, name='admin_download_global'),
#     path('admin/download-global/', views.download_global_model_admin, name='admin_download_global_latest'),
# ]

# models/urls.py
from django.urls import path
from . import views

app_name = 'models'

urlpatterns = [
    # Existing URLs
    path('upload/', views.upload_dataset, name='upload_dataset'),
    path('training/<str:session_id>/', views.training_progress, name='training_progress'),
    path('api/training-status/<str:session_id>/', views.api_training_status, name='api_training_status'),
    
    # NEW: Model download endpoints
    path('download/<str:session_id>/<str:model_type>/', views.download_model, name='download_model'),
    path('api/global-model-status/', views.check_global_model_availability, name='global_model_status'),
    
    # NEW: Debug endpoint
    path('debug/active-sessions/', views.debug_active_sessions, name='debug_active_sessions'),
]