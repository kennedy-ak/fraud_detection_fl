from django.shortcuts import render
from django.http import HttpResponse
from accounts.models import TrainingSession, ClientProfile
from django.db.models import Count, Avg

def landing_page(request):
    """Landing page with project statistics"""
    
    # Get some real statistics from your database
    context = {
        'total_clients': ClientProfile.objects.filter(is_active=True).count(),
        'total_sessions': TrainingSession.objects.count(),
        'completed_sessions': TrainingSession.objects.filter(status='completed').count(),
        'avg_accuracy': TrainingSession.objects.filter(
            status='completed', 
            accuracy__isnull=False
        ).aggregate(avg_acc=Avg('accuracy'))['avg_acc'] or 0,
    }
    
    # Format accuracy as percentage
    if context['avg_accuracy']:
        context['avg_accuracy_percent'] = round(context['avg_accuracy'] * 100, 1)
    else:
        context['avg_accuracy_percent'] = 0
    
    return render(request, 'main/landing.html', context)
