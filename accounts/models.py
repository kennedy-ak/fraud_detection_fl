# from django.db import models
# from django.contrib.auth.models import User

# class ClientProfile(models.Model):
#     user = models.OneToOneField(User, on_delete=models.CASCADE)
#     organization_name = models.CharField(max_length=200)
#     organization_type = models.CharField(max_length=100, choices=[
#         ('bank', 'Bank'),
#         ('fintech', 'Fintech Company'),
#         ('payment_processor', 'Payment Processor'),
#         ('other', 'Other')
#     ])
#     created_at = models.DateTimeField(auto_now_add=True)
#     is_active = models.BooleanField(default=True)
    
#     def __str__(self):
#         return f"{self.user.username} - {self.organization_name}"

# class TrainingSession(models.Model):
#     client = models.ForeignKey(ClientProfile, on_delete=models.CASCADE)
#     session_id = models.CharField(max_length=100, unique=True)
#     dataset_name = models.CharField(max_length=200)
#     created_at = models.DateTimeField(auto_now_add=True)
#     status = models.CharField(max_length=50, choices=[
#         ('pending', 'Pending'),
#         ('training', 'Training'),
#         ('completed', 'Completed'),
#         ('failed', 'Failed')
#     ], default='pending')
#     accuracy = models.FloatField(null=True, blank=True)
#     loss = models.FloatField(null=True, blank=True)
    
#     def __str__(self):
#         return f"{self.client.user.username} - {self.session_id}"

# accounts/models.py
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
    
    # Training results
    accuracy = models.FloatField(null=True, blank=True)
    loss = models.FloatField(null=True, blank=True)
    
    # NEW: Training parameters
    epochs = models.IntegerField(default=50, help_text="Number of training epochs")
    batch_size = models.IntegerField(default=32, help_text="Training batch size")
    
    # NEW: Training metrics
    training_time_seconds = models.FloatField(null=True, blank=True, help_text="Total training time in seconds")
    final_epoch = models.IntegerField(null=True, blank=True, help_text="Final epoch completed")
    
    def __str__(self):
        return f"{self.client.user.username} - {self.session_id}"
    
    def get_training_duration_display(self):
        """Return human-readable training duration"""
        if not self.training_time_seconds:
            return "Unknown"
        
        seconds = int(self.training_time_seconds)
        minutes = seconds // 60
        hours = minutes // 60
        
        if hours > 0:
            return f"{hours}h {minutes % 60}m {seconds % 60}s"
        elif minutes > 0:
            return f"{minutes}m {seconds % 60}s"
        else:
            return f"{seconds}s"
    
    def get_estimated_training_time(self):
        """Estimate training time based on epochs"""
        # Rough estimates based on typical dataset sizes
        base_time_per_epoch = 30  # seconds per epoch for average dataset
        
        if self.epochs <= 10:
            estimated_minutes = (self.epochs * base_time_per_epoch) // 60
            return f"~{max(1, estimated_minutes)}-{estimated_minutes + 2} minutes"
        elif self.epochs <= 50:
            estimated_minutes = (self.epochs * base_time_per_epoch) // 60
            return f"~{estimated_minutes}-{estimated_minutes + 5} minutes"
        else:
            estimated_minutes = (self.epochs * base_time_per_epoch) // 60
            return f"~{estimated_minutes}-{estimated_minutes + 10} minutes"