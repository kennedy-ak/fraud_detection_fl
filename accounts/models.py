from django.db import models
from django.contrib.auth.models import User

class ClientProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    organization_name = models.CharField(max_length=200)
    organization_type = models.CharField(max_length=100, choices=[
        ('bank', 'Bank'),
        ('fintech', 'Fintech Company'),
        ('payment_processor', 'Payment Processor'),
        ('other', 'Other')
    ])
    created_at = models.DateTimeField(auto_now_add=True)
    is_active = models.BooleanField(default=True)
    
    def __str__(self):
        return f"{self.user.username} - {self.organization_name}"

class TrainingSession(models.Model):
    client = models.ForeignKey(ClientProfile, on_delete=models.CASCADE)
    session_id = models.CharField(max_length=100, unique=True)
    dataset_name = models.CharField(max_length=200)
    created_at = models.DateTimeField(auto_now_add=True)
    status = models.CharField(max_length=50, choices=[
        ('pending', 'Pending'),
        ('training', 'Training'),
        ('completed', 'Completed'),
        ('failed', 'Failed')
    ], default='pending')
    accuracy = models.FloatField(null=True, blank=True)
    loss = models.FloatField(null=True, blank=True)
    
    def __str__(self):
        return f"{self.client.user.username} - {self.session_id}"