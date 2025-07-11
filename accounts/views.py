from django.shortcuts import render, redirect
from django.contrib.auth import login
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import JsonResponse
from django.db import models  # Add this import
from .forms import ClientRegistrationForm, DataUploadForm
from .models import ClientProfile, TrainingSession
import json

def register(request):
    if request.method == 'POST':
        form = ClientRegistrationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            messages.success(request, 'Registration successful!')
            return redirect('accounts:dashboard')
    else:
        form = ClientRegistrationForm()
    return render(request, 'accounts/register.html', {'form': form})

@login_required
def dashboard(request):
    try:
        client_profile = ClientProfile.objects.get(user=request.user)
    except ClientProfile.DoesNotExist:
        # Create profile if it doesn't exist
        client_profile = ClientProfile.objects.create(
            user=request.user,
            organization_name=f"{request.user.username} Organization",
            organization_type='other'
        )
    
    training_sessions = TrainingSession.objects.filter(client=client_profile).order_by('-created_at')
    
    # Calculate statistics
    total_sessions = training_sessions.count()
    completed_sessions = training_sessions.filter(status='completed').count()
    avg_accuracy = training_sessions.filter(status='completed').aggregate(
        avg_acc=models.Avg('accuracy')
    )['avg_acc'] or 0
    
    context = {
        'client_profile': client_profile,
        'training_sessions': training_sessions,
        'total_sessions': total_sessions,
        'completed_sessions': completed_sessions,
        'avg_accuracy': round(avg_accuracy, 4) if avg_accuracy else 0,
        'upload_form': DataUploadForm()
    }
    return render(request, 'accounts/dashboard.html', context)

@login_required
def api_training_progress(request):
    """API endpoint for real-time training progress"""
    try:
        client_profile = ClientProfile.objects.get(user=request.user)
    except ClientProfile.DoesNotExist:
        return JsonResponse({'sessions': []})
    
    sessions = TrainingSession.objects.filter(client=client_profile).order_by('-created_at')[:10]
    
    data = {
        'sessions': [
            {
                'id': session.id,
                'session_id': session.session_id,
                'status': session.status,
                'accuracy': session.accuracy,
                'loss': session.loss,
                'created_at': session.created_at.isoformat()
            }
            for session in sessions
        ]
    }
    return JsonResponse(data)


from django.contrib.auth import logout
from django.shortcuts import redirect
from django.contrib.auth.decorators import login_required

@login_required
def custom_logout(request):
    """Custom logout view that handles both GET and POST"""
    logout(request)
    messages.success(request, 'You have been successfully logged out.')
    return redirect('accounts:login')