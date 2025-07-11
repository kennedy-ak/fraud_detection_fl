# federated/urls.py
from django.urls import path
from . import views

app_name = 'federated'

urlpatterns = [
    # Admin dashboard
    path('admin/', views.admin_dashboard, name='admin_dashboard'),
    
    # Aggregation endpoints
    path('trigger-aggregation/', views.trigger_aggregation, name='trigger_aggregation'),
    
    # Global model download endpoints for admin
    path('admin/download-global/<int:round_id>/', views.download_global_model_admin, name='admin_download_global'),
    path('admin/download-global/', views.download_global_model_admin, name='admin_download_global_latest'),
    
    # API endpoints
    path('api/stats/', views.api_federated_stats, name='api_federated_stats'),
    
    # Debug endpoints
    path('debug/global-models/', views.debug_global_models, name='debug_global_models'),
]