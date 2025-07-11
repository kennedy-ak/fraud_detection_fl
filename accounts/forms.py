# from django import forms
# from django.contrib.auth.forms import UserCreationForm
# from django.contrib.auth.models import User
# from .models import ClientProfile

# class ClientRegistrationForm(UserCreationForm):
#     email = forms.EmailField(required=True)
#     organization_name = forms.CharField(max_length=200)
#     organization_type = forms.ChoiceField(choices=[
#         ('bank', 'Bank'),
#         ('fintech', 'Fintech Company'),
#         ('payment_processor', 'Payment Processor'),
#         ('other', 'Other')
#     ])
    
#     class Meta:
#         model = User
#         fields = ('username', 'email', 'password1', 'password2')
    
#     def save(self, commit=True):
#         user = super().save(commit=False)
#         user.email = self.cleaned_data['email']
#         if commit:
#             user.save()
#             ClientProfile.objects.create(
#                 user=user,
#                 organization_name=self.cleaned_data['organization_name'],
#                 organization_type=self.cleaned_data['organization_type']
#             )
#         return user

# class DataUploadForm(forms.Form):
#     dataset_file = forms.FileField(
#         help_text="""
#         Upload CSV file with transaction data. Required columns: amt, is_fraud.
#         Optional: lat, long, city_pop, unix_time, merch_lat, merch_long, category, gender, state, job.
#         Download sample template below.
#         """,
#         widget=forms.FileInput(attrs={'accept': '.csv'})
#     )
#     dataset_name = forms.CharField(max_length=200)

# accounts/forms.py
from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from .models import ClientProfile

class ClientRegistrationForm(UserCreationForm):
    email = forms.EmailField(required=True)
    organization_name = forms.CharField(max_length=200)
    organization_type = forms.ChoiceField(choices=[
        ('bank', 'Bank'),
        ('fintech', 'Fintech Company'),
        ('payment_processor', 'Payment Processor'),
        ('other', 'Other')
    ])
    
    class Meta:
        model = User
        fields = ('username', 'email', 'password1', 'password2')
    
    def save(self, commit=True):
        user = super().save(commit=False)
        user.email = self.cleaned_data['email']
        if commit:
            user.save()
            ClientProfile.objects.create(
                user=user,
                organization_name=self.cleaned_data['organization_name'],
                organization_type=self.cleaned_data['organization_type']
            )
        return user

class DataUploadForm(forms.Form):
    EPOCH_CHOICES = [
        (10, '10 epochs (Quick - 2-5 minutes)'),
        (25, '25 epochs (Balanced - 5-10 minutes)'),
        (50, '50 epochs (Standard - 10-20 minutes)'),
        (75, '75 epochs (Extended - 15-30 minutes)'),
        (100, '100 epochs (Comprehensive - 20-40 minutes)'),
    ]
    
    BATCH_SIZE_CHOICES = [
        (16, '16 (Small batches - slower but more stable)'),
        (32, '32 (Balanced - recommended)'),
        (64, '64 (Large batches - faster but needs more memory)'),
        (128, '128 (Very large - for powerful machines only)'),
    ]
    
    dataset_file = forms.FileField(
        help_text="Upload CSV file with transaction data (max 500MB)",
        widget=forms.FileInput(attrs={
            'accept': '.csv',
            'class': 'form-control'
        })
    )
    
    dataset_name = forms.CharField(
        max_length=200,
        help_text="Give your dataset a descriptive name",
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'e.g., Q4 2024 Credit Card Transactions'
        })
    )
    
    epochs = forms.ChoiceField(
        choices=EPOCH_CHOICES,
        initial=50,
        help_text="Number of training epochs - more epochs = better accuracy but longer training time",
        widget=forms.Select(attrs={
            'class': 'form-select'
        })
    )
    
    batch_size = forms.ChoiceField(
        choices=BATCH_SIZE_CHOICES,
        initial=32,
        help_text="Training batch size - affects memory usage and training speed",
        widget=forms.Select(attrs={
            'class': 'form-select'
        })
    )
    
    def clean_dataset_file(self):
        file = self.cleaned_data['dataset_file']
        
        # Check file size (500MB limit)
        if file.size > 500 * 1024 * 1024:
            raise forms.ValidationError("File size must be less than 500MB")
        
        # Check file extension
        if not file.name.lower().endswith('.csv'):
            raise forms.ValidationError("Please upload a CSV file")
        
        return file
    
    def clean_epochs(self):
        epochs = int(self.cleaned_data['epochs'])
        
        # Validate epoch range
        if epochs < 5:
            raise forms.ValidationError("Minimum 5 epochs required for meaningful training")
        if epochs > 200:
            raise forms.ValidationError("Maximum 200 epochs allowed to prevent excessive training time")
        
        return epochs
    
    def clean_batch_size(self):
        batch_size = int(self.cleaned_data['batch_size'])
        
        # Validate batch size
        if batch_size < 8:
            raise forms.ValidationError("Minimum batch size is 8")
        if batch_size > 256:
            raise forms.ValidationError("Maximum batch size is 256")
        
        return batch_size