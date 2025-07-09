# federated/aggregation.py
import torch
import os
import glob
from typing import List, Dict, Tuple
from .encryption import HomomorphicEncryption
from accounts.models import TrainingSession, ClientProfile
from models.ml_models import FraudDetectionModel
import json
import pickle
import time

class FederatedAggregator:
    def __init__(self):
        self.he = HomomorphicEncryption()
        self.global_model = None
        self.aggregation_history = []
        self.global_models_dir = 'media/models/global/'
        
        # Ensure global models directory exists
        os.makedirs(self.global_models_dir, exist_ok=True)
        
    def collect_client_updates(self) -> List[Dict]:
        """Collect encrypted model updates from all clients"""
        completed_sessions = TrainingSession.objects.filter(status='completed')
        
        client_updates = []
        for session in completed_sessions:
            weights_file = f'media/models/encrypted_weights_{session.session_id}.pkl'
            if os.path.exists(weights_file):
                encrypted_weights = self.he.load_encrypted_weights(weights_file)
                client_updates.append({
                    'session_id': session.session_id,
                    'client_id': session.client.id,
                    'client_name': session.client.organization_name,
                    'encrypted_weights': encrypted_weights,
                    'accuracy': session.accuracy
                })
        
        return client_updates
    
    def aggregate_models(self, client_updates: List[Dict]) -> Dict:
        """Perform federated averaging on encrypted models"""
        if not client_updates:
            return None
        
        print(f"[Aggregator] Starting aggregation with {len(client_updates)} client models")
        
        # Extract encrypted weights
        encrypted_weights_list = [update['encrypted_weights'] for update in client_updates]
        
        # Perform encrypted averaging - THE MAGIC HAPPENS HERE
        averaged_encrypted_weights = self.he.average_encrypted_weights(encrypted_weights_list)
        
        # Calculate global performance metrics
        total_accuracy = sum(update['accuracy'] for update in client_updates if update['accuracy'])
        num_clients = len([update for update in client_updates if update['accuracy']])
        avg_accuracy = total_accuracy / num_clients if num_clients > 0 else 0
        
        # Store aggregation result
        round_id = len(self.aggregation_history) + 1
        aggregation_result = {
            'round_id': round_id,
            'num_clients': len(client_updates),
            'participating_clients': [update['client_name'] for update in client_updates],
            'avg_accuracy': avg_accuracy,
            'encrypted_weights': averaged_encrypted_weights,
            'timestamp': time.time()
        }
        
        self.aggregation_history.append(aggregation_result)
        
        # CRITICAL: Save the global model to file system with proper serialization
        self._save_global_model(aggregation_result)
        
        print(f"[Aggregator] Global model round {round_id} created with accuracy {avg_accuracy:.4f}")
        
        return aggregation_result
    
    def _save_global_model(self, aggregation_result):
        """Save the global model to filesystem for client downloads with proper TenSEAL serialization"""
        round_id = aggregation_result['round_id']
        
        print(f"[Aggregator] Saving global model round {round_id}...")
        
        # CRITICAL FIX: Serialize TenSEAL encrypted weights properly
        try:
            serialized_weights = {}
            for name, encrypted_weight in aggregation_result['encrypted_weights'].items():
                # Use TenSEAL's built-in serialization
                serialized_weights[name] = encrypted_weight.serialize()
                print(f"[Aggregator] Serialized weight: {name}")
            
            print(f"[Aggregator] Successfully serialized {len(serialized_weights)} weight tensors")
            
        except Exception as e:
            print(f"[Aggregator] Error serializing encrypted weights: {e}")
            raise Exception(f"Failed to serialize encrypted weights: {e}")
        
        # Save encrypted global model with serialized weights
        global_model_path = os.path.join(self.global_models_dir, f'global_model_round_{round_id}.pkl')
        
        global_model_data = {
            'round_id': round_id,
            'num_clients': aggregation_result['num_clients'],
            'participating_clients': aggregation_result['participating_clients'],
            'global_accuracy': aggregation_result['avg_accuracy'],
            'serialized_encrypted_weights': serialized_weights,  # Use serialized version
            'timestamp': aggregation_result['timestamp'],
            'context_data': self.he.get_context_data(),  # Include context for deserialization
            'metadata': {
                'federation_type': 'homomorphic_encrypted',
                'algorithm': 'federated_averaging',
                'privacy_level': 'full_encryption',
                'tenseal_serialized': True  # Flag to indicate TenSEAL serialization
            }
        }
        
        # Save using pickle (now safe since we've serialized the TenSEAL objects)
        try:
            with open(global_model_path, 'wb') as f:
                pickle.dump(global_model_data, f)
            
            # Verify file was created and has content
            if os.path.exists(global_model_path):
                file_size = os.path.getsize(global_model_path)
                print(f"[Aggregator] Global model saved to {global_model_path} ({file_size} bytes)")
            else:
                raise Exception("File was not created")
                
        except Exception as e:
            print(f"[Aggregator] Error saving global model file: {e}")
            raise Exception(f"Failed to save global model: {e}")
        
        # Also save metadata as JSON for easy reading
        metadata_path = os.path.join(self.global_models_dir, f'global_model_round_{round_id}_metadata.json')
        try:
            with open(metadata_path, 'w') as f:
                json.dump({
                    'round_id': round_id,
                    'num_clients': aggregation_result['num_clients'],
                    'participating_clients': aggregation_result['participating_clients'],
                    'global_accuracy': aggregation_result['avg_accuracy'],
                    'timestamp': aggregation_result['timestamp'],
                    'file_path': global_model_path,
                    'weights_count': len(serialized_weights),
                    'serialization_method': 'tenseal_native'
                }, f, indent=2)
            
            print(f"[Aggregator] Metadata saved to {metadata_path}")
            
        except Exception as e:
            print(f"[Aggregator] Warning: Could not save metadata: {e}")
            # Don't fail aggregation if metadata save fails
    
    def load_global_model(self, round_id: int) -> Dict:
        """Load a global model and deserialize TenSEAL weights"""
        global_model_path = os.path.join(self.global_models_dir, f'global_model_round_{round_id}.pkl')
        
        if not os.path.exists(global_model_path):
            raise Exception(f"Global model round {round_id} not found")
        
        with open(global_model_path, 'rb') as f:
            global_model_data = pickle.load(f)
        
        # Check if this uses TenSEAL serialization
        if global_model_data.get('metadata', {}).get('tenseal_serialized', False):
            # Deserialize TenSEAL weights
            try:
                import tenseal as ts
                
                # Load context
                context_data = global_model_data['context_data']
                context = ts.context_from(context_data)
                
                # Deserialize weights
                encrypted_weights = {}
                for name, serialized_weight in global_model_data['serialized_encrypted_weights'].items():
                    encrypted_weights[name] = ts.ckks_vector_from(context, serialized_weight)
                
                global_model_data['encrypted_weights'] = encrypted_weights
                print(f"[Aggregator] Successfully deserialized {len(encrypted_weights)} weight tensors")
                
            except Exception as e:
                print(f"[Aggregator] Error deserializing TenSEAL weights: {e}")
                raise Exception(f"Failed to deserialize encrypted weights: {e}")
        
        return global_model_data
    
    def simulate_global_model_update(self, aggregation_result: Dict) -> Dict:
        """Simulate updating the global model with aggregated weights"""
        # In a real scenario, this would decrypt weights and update the global model
        # For simulation, we'll create a mock global model performance
        
        global_model_metrics = {
            'global_accuracy': aggregation_result['avg_accuracy'],
            'round_id': aggregation_result['round_id'],
            'participating_clients': aggregation_result['num_clients'],
            'improvement': 0.01 if len(self.aggregation_history) > 1 else 0  # Mock improvement
        }
        
        return global_model_metrics
    
    def get_aggregation_stats(self) -> Dict:
        """Get statistics about federated learning process"""
        if not self.aggregation_history:
            return {
                'total_rounds': 0,
                'best_accuracy': 0,
                'latest_accuracy': 0,
                'participating_clients': 0,
                'global_model_available': False
            }
        
        accuracies = [round_data['avg_accuracy'] for round_data in self.aggregation_history]
        
        return {
            'total_rounds': len(self.aggregation_history),
            'best_accuracy': max(accuracies),
            'latest_accuracy': accuracies[-1],
            'participating_clients': self.aggregation_history[-1]['num_clients'],
            'accuracy_history': accuracies,
            'global_model_available': True,
            'latest_round': self.aggregation_history[-1]['round_id']
        }