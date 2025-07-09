# models/views.py
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import JsonResponse, HttpResponse, Http404
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.db import transaction
from accounts.models import ClientProfile, TrainingSession
from accounts.forms import DataUploadForm
from .ml_models import FraudDetectionTrainer
import os
import uuid
import json
import threading
import time
import traceback

# Global dictionary to track running training sessions
ACTIVE_TRAINING_SESSIONS = {}
TRAINING_LOCK = threading.Lock()

@login_required
def upload_dataset(request):
    if request.method == 'POST':
        form = DataUploadForm(request.POST, request.FILES)
        if form.is_valid():
            file = request.FILES['dataset_file']
            
            # Generate a new unique session ID for each upload
            new_session_uuid = str(uuid.uuid4())
            file_name = f"{new_session_uuid}_{file.name}"
            file_path = default_storage.save(f'datasets/{file_name}', ContentFile(file.read()))
            
            try:
                with transaction.atomic():
                    client_profile = ClientProfile.objects.select_for_update().get(user=request.user)
                    
                    # Check for any active training sessions for this user
                    active_sessions = TrainingSession.objects.filter(
                        client=client_profile, 
                        status__in=['pending', 'training']
                    )
                    
                    if active_sessions.exists():
                        messages.warning(request, 'You already have an active training session. Please wait for it to complete.')
                        return redirect('accounts:dashboard')

                    # Double-check global tracking
                    with TRAINING_LOCK:
                        user_has_active_session = any(
                            session_info.get('user_id') == request.user.id 
                            for session_info in ACTIVE_TRAINING_SESSIONS.values()
                        )
                        
                        if user_has_active_session:
                            messages.warning(request, 'Training session already in progress for your account.')
                            return redirect('accounts:dashboard')

                        # Create new session
                        session = TrainingSession.objects.create(
                            client=client_profile,
                            session_id=new_session_uuid,
                            dataset_name=form.cleaned_data['dataset_name'],
                            status='training'  # Start directly as training
                        )
                        
                        # Add to global tracking with user info
                        ACTIVE_TRAINING_SESSIONS[session.session_id] = {
                            'user_id': request.user.id,
                            'started_at': time.time(),
                            'status': 'training'
                        }
                        
                        print(f"[Upload] Session {session.session_id} created and tracked globally.")

                # Start training in background
                threading.Thread(
                    target=train_model_background,
                    args=(session.id, file_path, session.session_id),
                    daemon=True,
                    name=f"Training-{session.session_id[:8]}"
                ).start()
                
                messages.success(request, 'Dataset uploaded successfully! Training started.')
                return redirect('accounts:dashboard')

            except Exception as e:
                print(f"[Upload] Error initiating training session: {e}")
                messages.error(request, 'Error initiating training. Please try again.')
                
                # Clean up on error
                with TRAINING_LOCK:
                    if new_session_uuid in ACTIVE_TRAINING_SESSIONS:
                        del ACTIVE_TRAINING_SESSIONS[new_session_uuid]
                
                return redirect('accounts:dashboard')
        else:
            messages.error(request, 'Invalid form submission. Please check your file.')
            return render(request, 'models/upload.html', {'form': form})
    else:
        form = DataUploadForm()
    
    return render(request, 'models/upload.html', {'form': form})

def train_model_background(session_id, file_path, session_uuid):
    """Background task for model training with GUARANTEED SINGLE EXECUTION"""
    session = None
    start_time = time.time()
    TIMEOUT_MINUTES = 45  # Increased timeout
    
    try:
        print(f"[Training] ===== STARTING BACKGROUND TRAINING =====")
        print(f"[Training] Session ID: {session_id}")
        print(f"[Training] Session UUID: {session_uuid}")
        print(f"[Training] Thread: {threading.current_thread().name}")
        
        # CRITICAL: Verify session is still active and not completed
        with transaction.atomic():
            session = TrainingSession.objects.select_for_update().get(id=session_id)
            
            if session.status in ['completed', 'failed']:
                print(f"[Training] Session {session_uuid} already in terminal state: {session.status}")
                return
            
            # Ensure status is 'training'
            if session.status != 'training':
                session.status = 'training'
                session.save()
        
        # Verify global tracking
        with TRAINING_LOCK:
            if session_uuid not in ACTIVE_TRAINING_SESSIONS:
                print(f"[Training] Session {session_uuid} not in global tracking. Adding it.")
                ACTIVE_TRAINING_SESSIONS[session_uuid] = {
                    'user_id': session.client.user.id,
                    'started_at': start_time,
                    'status': 'training'
                }
        
        print(f"[Training] Session verified and ready for training")
        
        # Initialize trainer - CRITICAL: New instance each time
        trainer = FraudDetectionTrainer()
        
        # Get full file path
        full_path = default_storage.path(file_path)
        print(f"[Training] Training with file: {full_path}")
        
        # Check timeout before training
        if time.time() - start_time > TIMEOUT_MINUTES * 60:
            raise TimeoutError("Training timeout reached before starting")
        
        # CRITICAL: Train model with timeout monitoring
        print(f"[Training] ===== CALLING trainer.train_model() =====")
        
        model, history, metrics = trainer.train_model(full_path, epochs=50)
        
        # CRITICAL: Verify training actually completed
        if model is None or history is None or metrics is None:
            raise Exception("trainer.train_model() returned None values - training failed")
        
        if not trainer.is_training_completed():
            raise Exception("Trainer reports training not completed")
        
        print(f"[Training] ===== TRAINING COMPLETED SUCCESSFULLY =====")
        print(f"[Training] Model training completed! Metrics: {metrics}")
        
        # Save model files
        try:
            trainer.save_model(model, session.session_id)
            print(f"[Training] Model saved successfully")
        except Exception as e:
            print(f"[Training] Model save warning: {e}")
        
        # Try encryption (optional)
        try:
            from federated.encryption import HomomorphicEncryption
            he = HomomorphicEncryption()
            model_weights = trainer.get_model_weights()
            if model_weights:
                encrypted_weights = he.encrypt_weights(model_weights)
                os.makedirs('media/models', exist_ok=True)
                weights_path = f'media/models/encrypted_weights_{session.session_id}.pkl'
                he.save_encrypted_weights(encrypted_weights, weights_path)
                print(f"[Training] Model weights encrypted and saved")
        except ImportError:
            print("[Training] HomomorphicEncryption not available. Skipping encryption.")
        except Exception as e:
            print(f"[Training] Encryption warning (non-critical): {e}")
        
        # CRITICAL: Update session to completed - ATOMIC OPERATION
        with transaction.atomic():
            session = TrainingSession.objects.select_for_update().get(id=session_id)
            
            # Double-check it's not already completed by another process
            if session.status == 'completed':
                print(f"[Training] Session {session_uuid} already marked as completed")
                return
            
            session.status = 'completed'
            session.accuracy = float(metrics.get('accuracy', 0.0))
            
            final_loss = 0.0
            if history and 'loss' in history and history['loss']:
                final_loss = float(history['loss'][-1])
            session.loss = final_loss
            
            session.save()
        
        print(f"[Training] ===== SESSION MARKED AS COMPLETED IN DATABASE =====")
        print(f"[Training] Final accuracy: {session.accuracy}")
        print(f"[Training] Final loss: {session.loss}")
        print(f"[Training] Total time: {time.time() - start_time:.2f} seconds")
            
    except TimeoutError as te:
        print(f"[Training] TIMEOUT ERROR: {te}")
        _mark_session_failed(session_id, session_uuid, "Training timeout")
            
    except Exception as e:
        error_msg = str(e)
        training_time = time.time() - start_time
        print(f"[Training] GENERAL ERROR: {error_msg}")
        print(f"[Training] Training time before error: {training_time:.2f} seconds")
        print(f"[Training] Full traceback: {traceback.format_exc()}")
        
        _mark_session_failed(session_id, session_uuid, error_msg)
    
    finally:
        # ALWAYS remove from global tracking
        with TRAINING_LOCK:
            if session_uuid in ACTIVE_TRAINING_SESSIONS:
                del ACTIVE_TRAINING_SESSIONS[session_uuid]
                print(f"[Training] Removed session {session_uuid} from global tracking")
        
        # Force cleanup
        import gc
        gc.collect()
        print(f"[Training] ===== BACKGROUND TRAINING FUNCTION COMPLETED =====")

def _mark_session_failed(session_id, session_uuid, error_msg):
    """Helper function to mark a session as failed"""
    try:
        with transaction.atomic():
            session = TrainingSession.objects.select_for_update().get(id=session_id)
            if session.status not in ['completed', 'failed']:  # Don't overwrite completed
                session.status = 'failed'
                session.save()
        print(f"[Training] Session {session_uuid} marked as FAILED: {error_msg}")
    except Exception as save_error:
        print(f"[Training] Could not update failed status: {save_error}")

@login_required
def training_progress(request, session_id):
    """View training progress"""
    try:
        client_profile = ClientProfile.objects.get(user=request.user)
        session = TrainingSession.objects.get(
            session_id=session_id, 
            client=client_profile
        )
        
        print(f"[View] Training progress for {session_id}: status={session.status}")
        
        context = {
            'session': session,
            'progress_data': {
                'status': session.status,
                'accuracy': session.accuracy,
                'loss': session.loss
            }
        }
        
        return render(request, 'models/training.html', context)
    except TrainingSession.DoesNotExist:
        messages.error(request, 'Training session not found or you do not have permission to view it.')
        return redirect('accounts:dashboard')

@login_required
def api_training_status(request, session_id):
    """API endpoint for real-time training status"""
    try:
        client_profile = ClientProfile.objects.get(user=request.user)
        session = TrainingSession.objects.get(
            session_id=session_id,
            client=client_profile
        )
        
        print(f"[API] Status check for {session_id}: {session.status} (accuracy: {session.accuracy})")
        
        # Check if it's in global tracking
        with TRAINING_LOCK:
            is_globally_active = session_id in ACTIVE_TRAINING_SESSIONS
        
        response_data = {
            'status': session.status,
            'accuracy': session.accuracy,
            'loss': session.loss,
            'created_at': session.created_at.isoformat(),
            'session_id': session.session_id,
            'is_active_in_memory': is_globally_active
        }
        
        return JsonResponse(response_data)
        
    except ClientProfile.DoesNotExist:
        return JsonResponse({'error': 'Client profile not found'}, status=404)
    except TrainingSession.DoesNotExist:
        return JsonResponse({'error': 'Session not found or forbidden'}, status=404)
    except Exception as e:
        print(f"[API] Error in status check: {e}")
        return JsonResponse({'error': 'Internal server error'}, status=500)

# ===== MODEL DOWNLOAD FUNCTIONALITY =====

@login_required
def download_model(request, session_id, model_type='local'):
    """Download either local or global model based on model_type parameter"""
    try:
        client_profile = ClientProfile.objects.get(user=request.user)
        session = TrainingSession.objects.get(
            session_id=session_id,
            client=client_profile
        )
        
        if session.status != 'completed':
            messages.error(request, 'Training session not completed yet.')
            return redirect('models:training_progress', session_id=session_id)
        
        if model_type == 'local':
            return _download_local_model(session)
        elif model_type == 'global':
            return _download_global_model(session)
        else:
            raise Http404("Invalid model type")
            
    except TrainingSession.DoesNotExist:
        messages.error(request, 'Training session not found.')
        return redirect('accounts:dashboard')
    except Exception as e:
        messages.error(request, f'Download failed: {str(e)}')
        return redirect('models:training_progress', session_id=session_id)

def _download_local_model(session):
    """Download the local encrypted model weights"""
    local_weights_path = f'media/models/encrypted_weights_{session.session_id}.pkl'
    
    if not os.path.exists(local_weights_path):
        raise Exception("Local model weights not found")
    
    with open(local_weights_path, 'rb') as f:
        response = HttpResponse(f.read(), content_type='application/octet-stream')
        response['Content-Disposition'] = f'attachment; filename="local_model_{session.session_id}.pkl"'
        return response

def _download_global_model(session):
    """Download the latest global federated model with improved file detection and TenSEAL support"""
    from django.conf import settings
    
    # Use absolute path to avoid path resolution issues
    global_models_dir = os.path.join(settings.BASE_DIR, 'media', 'models', 'global')
    
    print(f"[GlobalDownload] Looking for global models in: {global_models_dir}")
    print(f"[GlobalDownload] Directory exists: {os.path.exists(global_models_dir)}")
    
    if not os.path.exists(global_models_dir):
        raise Exception("No global models directory found. Admin needs to run aggregation first.")
    
    # Get all global model files
    try:
        all_files = os.listdir(global_models_dir)
        global_model_files = [f for f in all_files if f.startswith('global_model_round_') and f.endswith('.pkl')]
        
        print(f"[GlobalDownload] All files in directory: {all_files}")
        print(f"[GlobalDownload] Global model files found: {global_model_files}")
        
    except Exception as e:
        print(f"[GlobalDownload] Error listing directory: {e}")
        raise Exception(f"Error accessing global models directory: {e}")
    
    if not global_model_files:
        raise Exception("No global models available yet. Admin needs to run aggregation first.")
    
    # Sort by round number to get the latest (more robust sorting)
    def extract_round_number(filename):
        try:
            # Extract number from 'global_model_round_X.pkl'
            return int(filename.split('_')[-1].split('.')[0])
        except:
            return 0
    
    global_model_files.sort(key=extract_round_number)
    latest_global_model = global_model_files[-1]
    latest_round = extract_round_number(latest_global_model)
    
    print(f"[GlobalDownload] Latest global model: {latest_global_model} (Round {latest_round})")
    
    global_model_path = os.path.join(global_models_dir, latest_global_model)
    
    # Verify file exists and is readable
    if not os.path.exists(global_model_path):
        raise Exception(f"Global model file not found: {global_model_path}")
    
    if not os.access(global_model_path, os.R_OK):
        raise Exception(f"Global model file not readable: {global_model_path}")
    
    file_size = os.path.getsize(global_model_path)
    print(f"[GlobalDownload] File size: {file_size} bytes")
    
    if file_size == 0:
        raise Exception("Global model file is empty")
    
    # Verify the file can be loaded (optional validation)
    try:
        import pickle
        with open(global_model_path, 'rb') as f:
            global_model_data = pickle.load(f)
        
        # Check if it's a properly formatted global model
        required_keys = ['round_id', 'global_accuracy', 'metadata']
        missing_keys = [key for key in required_keys if key not in global_model_data]
        
        if missing_keys:
            print(f"[GlobalDownload] Warning: Missing keys in global model: {missing_keys}")
        else:
            print(f"[GlobalDownload] Global model validation successful")
            print(f"[GlobalDownload] Round: {global_model_data['round_id']}, Accuracy: {global_model_data['global_accuracy']:.4f}")
            
    except Exception as e:
        print(f"[GlobalDownload] Warning: Could not validate global model file: {e}")
        # Continue with download even if validation fails
    
    # Return file response
    try:
        with open(global_model_path, 'rb') as f:
            response = HttpResponse(f.read(), content_type='application/octet-stream')
            response['Content-Disposition'] = f'attachment; filename="{latest_global_model}"'
            response['Content-Length'] = file_size
            
            print(f"[GlobalDownload] Successfully serving {latest_global_model}")
            return response
            
    except Exception as e:
        print(f"[GlobalDownload] Error reading file: {e}")
        raise Exception(f"Error reading global model file: {e}")

@login_required
def check_global_model_availability(request):
    """API endpoint to check if global model is available - with better error handling"""
    from django.conf import settings
    
    try:
        global_models_dir = os.path.join(settings.BASE_DIR, 'media', 'models', 'global')
        
        print(f"[GlobalModelCheck] Checking directory: {global_models_dir}")
        
        if not os.path.exists(global_models_dir):
            return JsonResponse({
                'available': False, 
                'message': 'No global models directory found',
                'debug_path': global_models_dir
            })
        
        all_files = os.listdir(global_models_dir)
        global_model_files = [f for f in all_files if f.startswith('global_model_round_') and f.endswith('.pkl')]
        
        print(f"[GlobalModelCheck] Files found: {len(global_model_files)} global models")
        
        if not global_model_files:
            return JsonResponse({
                'available': False, 
                'message': 'No global models yet',
                'debug_all_files': all_files,
                'debug_path': global_models_dir
            })
        
        # Get latest round info
        def extract_round_number(filename):
            try:
                return int(filename.split('_')[-1].split('.')[0])
            except:
                return 0
        
        global_model_files.sort(key=extract_round_number)
        latest_round = extract_round_number(global_model_files[-1])
        
        # Get file info
        latest_file_path = os.path.join(global_models_dir, global_model_files[-1])
        file_size = os.path.getsize(latest_file_path)
        file_time = os.path.getmtime(latest_file_path)
        
        return JsonResponse({
            'available': True,
            'latest_round': latest_round,
            'total_rounds': len(global_model_files),
            'message': f'Global model round {latest_round} available',
            'file_size_mb': round(file_size / (1024 * 1024), 2),
            'created_timestamp': file_time,
            'debug_latest_file': global_model_files[-1]
        })
        
    except Exception as e:
        print(f"[GlobalModelCheck] Error: {e}")
        return JsonResponse({
            'available': False, 
            'message': str(e),
            'error': True
        })

@login_required
def debug_active_sessions(request):
    """Debug endpoint to see active sessions"""
    if not request.user.is_superuser:
        return JsonResponse({'error': 'Unauthorized'}, status=403)
    
    with TRAINING_LOCK:
        active_sessions = dict(ACTIVE_TRAINING_SESSIONS)
    
    return JsonResponse({
        'active_sessions': active_sessions,
        'count': len(active_sessions)
    })