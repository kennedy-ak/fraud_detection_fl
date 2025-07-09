# federated/views.py
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required, user_passes_test
from django.http import JsonResponse, HttpResponse
from django.contrib import messages
from django.db import models
from django.conf import settings
from .aggregation import FederatedAggregator
from .encryption import HomomorphicEncryption
from accounts.models import TrainingSession, ClientProfile
import json
import os
import traceback

def is_admin(user):
    return user.is_superuser

@login_required
@user_passes_test(is_admin)
def admin_dashboard(request):
    """Admin dashboard for monitoring federated learning"""
    aggregator = FederatedAggregator()
    
    # Get all client statistics
    all_clients = ClientProfile.objects.all()
    client_stats = []
    
    for client in all_clients:
        sessions = TrainingSession.objects.filter(client=client)
        completed_sessions = sessions.filter(status='completed')
        
        client_stats.append({
            'client': client,
            'total_sessions': sessions.count(),
            'completed_sessions': completed_sessions.count(),
            'avg_accuracy': completed_sessions.aggregate(
                avg_acc=models.Avg('accuracy')
            )['avg_acc'] or 0,
            'last_training': sessions.order_by('-created_at').first()
        })
    
    # Get aggregation statistics
    aggregation_stats = aggregator.get_aggregation_stats()
    
    # Get global model files info
    global_models_dir = os.path.join(settings.BASE_DIR, 'media', 'models', 'global')
    global_models = []
    if os.path.exists(global_models_dir):
        # Only get .pkl files, not .json metadata files
        global_model_files = [f for f in os.listdir(global_models_dir) 
                             if f.startswith('global_model_round_') and f.endswith('.pkl')]
        global_model_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        
        for model_file in global_model_files:
            round_id = int(model_file.split('_')[-1].split('.')[0])
            file_path = os.path.join(global_models_dir, model_file)
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
            
            global_models.append({
                'round_id': round_id,
                'filename': model_file,
                'size_mb': round(file_size, 2),
                'created': os.path.getctime(file_path)
            })
    
    context = {
        'client_stats': client_stats,
        'aggregation_stats': aggregation_stats,
        'total_clients': all_clients.count(),
        'active_clients': all_clients.filter(is_active=True).count(),
        'global_models': global_models
    }
    
    return render(request, 'federated/admin_dashboard.html', context)

@login_required
@user_passes_test(is_admin)
def trigger_aggregation(request):
    """Enhanced aggregation with global model creation and detailed logging"""
    if request.method == 'POST':
        try:
            print(f"[Admin] ===== AGGREGATION TRIGGERED =====")
            print(f"[Admin] User: {request.user.username}")
            print(f"[Admin] Time: {request.META.get('HTTP_X_FORWARDED_FOR', request.META.get('REMOTE_ADDR'))}")
            
            aggregator = FederatedAggregator()
            
            # Step 1: Collect client updates
            print(f"[Admin] Step 1: Collecting client updates...")
            client_updates = aggregator.collect_client_updates()
            
            print(f"[Admin] Found {len(client_updates)} client updates")
            for update in client_updates:
                print(f"[Admin] - Client: {update['client_name']}, Accuracy: {update['accuracy']}")
            
            if not client_updates:
                print(f"[Admin] No client updates available for aggregation")
                messages.warning(request, 'No client updates available for aggregation.')
                return JsonResponse({
                    'status': 'warning', 
                    'message': 'No completed training sessions found. Clients need to complete training first.',
                    'participating_clients': 0
                })
            
            # Step 2: Perform aggregation and create global model
            print(f"[Admin] Step 2: Performing federated aggregation...")
            aggregation_result = aggregator.aggregate_models(client_updates)
            
            if aggregation_result:
                print(f"[Admin] ===== AGGREGATION SUCCESSFUL =====")
                print(f"[Admin] Round ID: {aggregation_result['round_id']}")
                print(f"[Admin] Global Accuracy: {aggregation_result['avg_accuracy']:.4f}")
                print(f"[Admin] Participating Clients: {len(client_updates)}")
                print(f"[Admin] Client Names: {aggregation_result['participating_clients']}")
                
                # Verify global model file was created
                global_models_dir = os.path.join(settings.BASE_DIR, 'media', 'models', 'global')
                expected_file = os.path.join(global_models_dir, f"global_model_round_{aggregation_result['round_id']}.pkl")
                
                if os.path.exists(expected_file):
                    file_size = os.path.getsize(expected_file)
                    print(f"[Admin] Global model file created: {expected_file} ({file_size} bytes)")
                else:
                    print(f"[Admin] WARNING: Global model file not found: {expected_file}")
                
                messages.success(request, 
                    f'Aggregation completed! Global model round {aggregation_result["round_id"]} '
                    f'created with accuracy: {aggregation_result["avg_accuracy"]:.4f}'
                )
                
                return JsonResponse({
                    'status': 'success',
                    'round_id': aggregation_result['round_id'],
                    'global_accuracy': aggregation_result['avg_accuracy'],
                    'avg_accuracy': aggregation_result['avg_accuracy'],  # For backward compatibility
                    'participating_clients': len(client_updates),
                    'num_clients': len(client_updates),  # For backward compatibility
                    'client_names': aggregation_result['participating_clients'],
                    'message': f'Global model round {aggregation_result["round_id"]} is now available for download',
                    'file_created': os.path.exists(expected_file) if 'expected_file' in locals() else False
                })
            else:
                raise Exception("Aggregation failed to produce results")
            
        except Exception as e:
            error_msg = str(e)
            print(f"[Admin] ===== AGGREGATION FAILED =====")
            print(f"[Admin] Error: {error_msg}")
            print(f"[Admin] Traceback: {traceback.format_exc()}")
            
            messages.error(request, f'Aggregation failed: {error_msg}')
            return JsonResponse({
                'status': 'error', 
                'message': error_msg,
                'participating_clients': 0
            })
    
    return JsonResponse({
        'status': 'error', 
        'message': 'Invalid request method. Use POST.',
        'participating_clients': 0
    })

@login_required
@user_passes_test(is_admin)
def download_global_model_admin(request, round_id=None):
    """Admin endpoint to download specific global model rounds"""
    try:
        global_models_dir = os.path.join(settings.BASE_DIR, 'media', 'models', 'global')
        
        if round_id:
            model_file = f'global_model_round_{round_id}.pkl'
        else:
            # Get latest model
            if not os.path.exists(global_models_dir):
                messages.error(request, 'No global models available')
                return redirect('federated:admin_dashboard')
                
            global_model_files = [f for f in os.listdir(global_models_dir) if f.startswith('global_model_round_')]
            if not global_model_files:
                messages.error(request, 'No global models available')
                return redirect('federated:admin_dashboard')
            
            global_model_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
            model_file = global_model_files[-1]
        
        model_path = os.path.join(global_models_dir, model_file)
        
        if not os.path.exists(model_path):
            messages.error(request, f'Global model round {round_id} not found')
            return redirect('federated:admin_dashboard')
        
        with open(model_path, 'rb') as f:
            response = HttpResponse(f.read(), content_type='application/octet-stream')
            response['Content-Disposition'] = f'attachment; filename="{model_file}"'
            return response
            
    except Exception as e:
        messages.error(request, f'Download failed: {str(e)}')
        return redirect('federated:admin_dashboard')

@login_required
def api_federated_stats(request):
    """API endpoint for federated learning statistics"""
    aggregator = FederatedAggregator()
    stats = aggregator.get_aggregation_stats()
    
    # Add client participation data
    all_sessions = TrainingSession.objects.filter(status='completed')
    client_participation = {}
    
    for session in all_sessions:
        client_name = session.client.organization_name
        if client_name not in client_participation:
            client_participation[client_name] = 0
        client_participation[client_name] += 1
    
    stats['client_participation'] = client_participation
    
    return JsonResponse(stats)

# Debug endpoint for checking global model status
@login_required
@user_passes_test(is_admin)
def debug_global_models(request):
    """Debug endpoint to check global model directory status"""
    global_models_dir = os.path.join(settings.BASE_DIR, 'media', 'models', 'global')
    
    debug_info = {
        'directory_path': global_models_dir,
        'directory_exists': os.path.exists(global_models_dir),
        'files': [],
        'permissions': {
            'readable': os.access(global_models_dir, os.R_OK) if os.path.exists(global_models_dir) else False,
            'writable': os.access(global_models_dir, os.W_OK) if os.path.exists(global_models_dir) else False,
        }
    }
    
    if os.path.exists(global_models_dir):
        try:
            all_files = os.listdir(global_models_dir)
            for file in all_files:
                file_path = os.path.join(global_models_dir, file)
                debug_info['files'].append({
                    'name': file,
                    'size': os.path.getsize(file_path),
                    'created': os.path.getctime(file_path),
                    'modified': os.path.getmtime(file_path),
                    'is_global_model': file.startswith('global_model_round_')
                })
        except Exception as e:
            debug_info['error'] = str(e)
    
    return JsonResponse(debug_info, indent=2)