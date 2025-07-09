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
    dataset_file = forms.FileField(
        help_text="""
        Upload CSV file with transaction data. Required columns: amt, is_fraud.
        Optional: lat, long, city_pop, unix_time, merch_lat, merch_long, category, gender, state, job.
        Download sample template below.
        """,
        widget=forms.FileInput(attrs={'accept': '.csv'})
    )
    dataset_name = forms.CharField(max_length=200)