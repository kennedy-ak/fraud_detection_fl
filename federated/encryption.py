import tenseal as ts
import torch
import pickle
import os
from typing import Dict, List

class HomomorphicEncryption:
    def __init__(self):
        # Create TenSEAL context
        self.context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes=[60, 40, 40, 60]
        )
        self.context.generate_galois_keys()
        self.context.global_scale = 2**40
        
    def encrypt_weights(self, weights: Dict[str, torch.Tensor]) -> Dict[str, ts.CKKSVector]:
        """Encrypt model weights using homomorphic encryption"""
        encrypted_weights = {}
        
        for name, weight in weights.items():
            # Flatten weight tensor and convert to list
            weight_flat = weight.flatten().tolist()
            
            # Encrypt the weight vector
            encrypted_weight = ts.ckks_vector(self.context, weight_flat)
            encrypted_weights[name] = encrypted_weight
            
        return encrypted_weights
    
    def decrypt_weights(self, encrypted_weights: Dict[str, ts.CKKSVector], 
                       original_shapes: Dict[str, tuple]) -> Dict[str, torch.Tensor]:
        """Decrypt encrypted weights back to PyTorch tensors"""
        decrypted_weights = {}
        
        for name, encrypted_weight in encrypted_weights.items():
            # Decrypt the vector
            decrypted_flat = encrypted_weight.decrypt()
            
            # Reshape back to original shape
            original_shape = original_shapes[name]
            decrypted_tensor = torch.tensor(decrypted_flat).reshape(original_shape)
            decrypted_weights[name] = decrypted_tensor
            
        return decrypted_weights
    
    def add_encrypted_weights(self, encrypted_weights_list: List[Dict[str, ts.CKKSVector]]) -> Dict[str, ts.CKKSVector]:
        """Add multiple encrypted weight dictionaries (for federated averaging)"""
        if not encrypted_weights_list:
            return {}
        
        # Initialize result with first set of weights
        result = encrypted_weights_list[0].copy()
        
        # Add remaining weight sets
        for weights in encrypted_weights_list[1:]:
            for name in result.keys():
                result[name] += weights[name]
        
        return result
    
    def average_encrypted_weights(self, encrypted_weights_list: List[Dict[str, ts.CKKSVector]]) -> Dict[str, ts.CKKSVector]:
        """Average multiple encrypted weight dictionaries"""
        if not encrypted_weights_list:
            return {}
        
        # Sum all weights
        summed_weights = self.add_encrypted_weights(encrypted_weights_list)
        
        # Divide by number of clients
        num_clients = len(encrypted_weights_list)
        averaged_weights = {}
        
        for name, weight in summed_weights.items():
            averaged_weights[name] = weight * (1.0 / num_clients)
        
        return averaged_weights
    
    def save_encrypted_weights(self, encrypted_weights: Dict[str, ts.CKKSVector], file_path: str):
        """Save encrypted weights to file"""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Convert to serializable format
        serializable_weights = {}
        for name, weight in encrypted_weights.items():
            serializable_weights[name] = weight.serialize()
        
        with open(file_path, 'wb') as f:
            pickle.dump(serializable_weights, f)
    
    def load_encrypted_weights(self, file_path: str) -> Dict[str, ts.CKKSVector]:
        """Load encrypted weights from file"""
        with open(file_path, 'rb') as f:
            serializable_weights = pickle.load(f)
        
        # Deserialize weights
        encrypted_weights = {}
        for name, serialized_weight in serializable_weights.items():
            encrypted_weights[name] = ts.ckks_vector_from(self.context, serialized_weight)
        
        return encrypted_weights
    
    def get_context_data(self) -> bytes:
        """Get context data for sharing with clients"""
        return self.context.serialize()
    
    def load_context_data(self, context_data: bytes):
        """Load context data from central server"""
        self.context = ts.context_from(context_data)