from django.contrib import admin
from .models import ClientProfile, TrainingSession

@admin.register(ClientProfile)
class ClientProfileAdmin(admin.ModelAdmin):
    list_display = ('user', 'organization_name', 'organization_type', 'created_at', 'is_active')
    list_filter = ('organization_type', 'is_active', 'created_at')
    search_fields = ('user__username', 'organization_name')
    readonly_fields = ('created_at',)

@admin.register(TrainingSession)
class TrainingSessionAdmin(admin.ModelAdmin):
    list_display = ('session_id', 'client', 'dataset_name', 'status', 'accuracy', 'created_at')
    list_filter = ('status', 'created_at')
    search_fields = ('session_id', 'dataset_name', 'client__organization_name')
    readonly_fields = ('session_id', 'created_at')