import numpy as np
import hashlib
import time
import random
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
import json
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.backends import default_backend
from sympy import isprime, nextprime
from math import gcd


@dataclass
class EllipticCurveParams:
    """Elliptic curve parameters"""
    p: int  # Finite field order
    a: int  # Curve parameter a
    b: int  # Curve parameter b
    n: int  # Base point G order
    h: int  # Cofactor
    Gx: int  # Base point G x-coordinate
    Gy: int  # Base point G y-coordinate


class BilinearGroup:
    """
    Bilinear group implementation
    Supports Type A (symmetric) and Type III (asymmetric) pairings
    """

    def __init__(self, curve_type: str = "A", security_param: int = 128):
        """
        Initialize bilinear group

        Args:
            curve_type: Curve type ("A" symmetric or "III" asymmetric)
            security_param: Security parameter
        """
        self.curve_type = curve_type
        self.security_param = security_param
        self.pairing_count = 0  # Initialize pairing counter

        # Generate curve parameters
        self.params = self._generate_curve_params()

        # Initialize group elements
        self.G1 = self._init_group_G1()
        self.G2 = self._init_group_G2()
        self.GT = self._init_group_GT()
        self.Zr = self._init_group_Zr()

        # Precompute pairing table (for optimization)
        self.pairing_table = {}

        print(f"Initialized {curve_type}-type bilinear group, security parameter: {security_param}")

    def _generate_curve_params(self) -> EllipticCurveParams:
        """Generate elliptic curve parameters"""
        # Determine prime size based on security parameter
        bit_length = self.security_param * 2

        # Generate pairing-friendly prime
        p = self._find_pairing_friendly_prime(bit_length)

        # Simplified: Use known curve parameters
        if self.curve_type == "A":
            # Type A (symmetric) curve parameters
            return EllipticCurveParams(
                p=p,
                a=0,
                b=3,  # y² = x³ + 3
                n=p,  # Simplified assumption
                h=1,
                Gx=1,
                Gy=2  # Simplified assumption
            )
        else:
            # Type III (asymmetric) curve parameters
            return EllipticCurveParams(
                p=p,
                a=0,
                b=2,  # y² = x³ + 2
                n=p,  # Simplified assumption
                h=1,
                Gx=1,
                Gy=1  # Simplified assumption
            )

    def _find_pairing_friendly_prime(self, bit_length: int) -> int:
        """Find pairing-friendly prime"""
        # Simplified implementation: Find a sufficiently large prime
        start = 2 ** bit_length
        p = nextprime(start)

        # Ensure p ≡ 3 mod 4 (simplifies square root calculation)
        while p % 4 != 3:
            p = nextprime(p + 1)

        return p

    def _init_group_G1(self):
        """Initialize G1 group"""
        return {
            'params': self.params,
            'zero': self._point_at_infinity(),
            'gen': self._create_point(self.params.Gx, self.params.Gy)
        }

    def _init_group_G2(self):
        """Initialize G2 group"""
        if self.curve_type == "A":
            # In symmetric pairings, G2 = G1
            return self.G1
        else:
            # In asymmetric pairings, G2 is a different group
            return {
                'params': self.params,
                'zero': self._point_at_infinity(),
                'gen': self._create_point(self.params.Gx + 1, self.params.Gy + 1)
            }

    def _init_group_GT(self):
        """Initialize GT group (target group)"""
        # GT is a multiplicative group of order n
        return {
            'order': self.params.n,
            'one': 1,  # Identity element
            'gen': self._find_primitive_root(self.params.n)
        }

    def _init_group_Zr(self):
        """Initialize Zr group (integers mod n)"""
        return {
            'order': self.params.n,
            'zero': 0,
            'one': 1
        }

    def _point_at_infinity(self):
        """Return point at infinity"""
        return (None, None)

    def _create_point(self, x: int, y: int) -> Tuple[int, int]:
        """Create elliptic curve point"""
        # Verify point is on curve
        if not self._is_point_on_curve(x, y):
            raise ValueError(f"Point ({x}, {y}) is not on the curve")
        return (x, y)

    def _is_point_on_curve(self, x: int, y: int) -> bool:
        """Verify if point is on elliptic curve"""
        if x is None and y is None:  # Point at infinity
            return True

        left = (y * y) % self.params.p
        right = (x * x * x + self.params.a * x + self.params.b) % self.params.p
        return left == right

    def _find_primitive_root(self, n: int) -> int:
        """Find primitive root modulo n"""
        # Simplified implementation
        for g in range(2, n):
            if gcd(g, n) == 1:
                return g
        return 2  # Default

    def add_points(self, P: Tuple[int, int], Q: Tuple[int, int]) -> Tuple[int, int]:
        """Elliptic curve point addition"""
        if P == self._point_at_infinity():
            return Q
        if Q == self._point_at_infinity():
            return P

        x1, y1 = P
        x2, y2 = Q

        if x1 == x2 and (y1 != y2 or y1 == 0):
            return self._point_at_infinity()

        if x1 == x2:
            # Point doubling
            s = (3 * x1 * x1 + self.params.a) * pow(2 * y1, -1, self.params.p) % self.params.p
        else:
            # Point addition
            s = (y2 - y1) * pow(x2 - x1, -1, self.params.p) % self.params.p

        x3 = (s * s - x1 - x2) % self.params.p
        y3 = (s * (x1 - x3) - y1) % self.params.p

        return (x3, y3)

    def scalar_mult(self, k: int, P: Tuple[int, int]) -> Tuple[int, int]:
        """Elliptic curve scalar multiplication"""
        if k == 0:
            return self._point_at_infinity()

        # Use double-and-add algorithm
        result = self._point_at_infinity()
        addend = P

        while k:
            if k & 1:
                result = self.add_points(result, addend)
            addend = self.add_points(addend, addend)
            k //= 2

        return result

    def pairing(self, P: Tuple[int, int], Q: Tuple[int, int]) -> int:
        """Bilinear pairing function"""
        # Increment pairing counter
        self.pairing_count += 1

        # Check if precomputed
        key = (hash(P), hash(Q))
        if key in self.pairing_table:
            return self.pairing_table[key]

        # Simplified pairing calculation
        if self.curve_type == "A":
            # Symmetric pairing simplification
            result = pow(self.GT['gen'], (P[0] * Q[0] + P[1] * Q[1]) % self.params.n, self.params.p)
        else:
            # Asymmetric pairing simplification
            result = pow(self.GT['gen'], (P[0] * Q[1] + P[1] * Q[0]) % self.params.n, self.params.p)

        # Cache result
        self.pairing_table[key] = result

        return result

    def random_element(self, group: str = "G1") -> Any:
        """Generate random group element"""
        if group == "G1":
            k = random.randint(1, self.params.n - 1)
            return self.scalar_mult(k, self.G1['gen'])
        elif group == "G2":
            k = random.randint(1, self.params.n - 1)
            return self.scalar_mult(k, self.G2['gen'])
        elif group == "GT":
            k = random.randint(1, self.params.n - 1)
            return pow(self.GT['gen'], k, self.params.p)
        elif group == "Zr":
            return random.randint(1, self.params.n - 1)
        else:
            raise ValueError(f"Unknown group: {group}")

    def is_in_group(self, element: Any, group: str = "G1") -> bool:
        """Verify if element is in specified group"""
        if group == "G1":
            return self._is_point_on_curve(element[0], element[1]) if element != self._point_at_infinity() else True
        elif group == "G2":
            return self._is_point_on_curve(element[0], element[1]) if element != self._point_at_infinity() else True
        elif group == "GT":
            return 1 <= element < self.params.p and gcd(element, self.params.p) == 1
        elif group == "Zr":
            return 0 <= element < self.params.n
        else:
            return False

    def reset_pairing_count(self):
        """Reset pairing counter"""
        self.pairing_count = 0


class QuadraticMIFE:
    """
    Quadratic Multi-Input Functional Encryption for Vertical Federated Learning
    Based on: Agrawal, Goyal, and Tomida [18] construction
    Adapted for SFedV protocol
    """

    def __init__(self, security_param: int = 128, curve_type: str = "A"):
        """
        Initialize qMIFE scheme

        Args:
            security_param: Security parameter
            curve_type: Curve type ("A" symmetric or "III" asymmetric)
        """
        self.security_param = security_param
        self.curve_type = curve_type

        # Initialize bilinear group
        self.bg = BilinearGroup(curve_type, security_param)
        self.order = self.bg.params.n

        # State for current iteration
        self.current_iteration = 0
        self.pp = None
        self.ek = {}
        self.msk = None

    def setup(self, n_slots: int):
        """
        Setup algorithm for qMIFE
        Generates new instance for each iteration to prevent mix-and-match attacks

        Args:
            n_slots: Number of encryption slots (clients)
        """
        # Reset pairing counter
        self.bg.reset_pairing_count()

        # Generate master secret key elements
        total_dim = n_slots + 1  # +1 for labels
        alpha = [self.bg.random_element("Zr") for _ in range(total_dim)]
        beta = [self.bg.random_element("Zr") for _ in range(total_dim)]
        gamma = [self.bg.random_element("Zr") for _ in range(total_dim)]

        # Generate group generators
        g1 = self.bg.G1['gen']
        g2 = self.bg.G2['gen']

        # Compute public parameters
        self.pp = {
            'g1': g1,
            'g2': g2,
            'h1_i': [self.bg.scalar_mult(alpha[i], g1) for i in range(total_dim)],
            'h2_i': [self.bg.scalar_mult(beta[i], g2) for i in range(total_dim)],
            'h3_i': [self.bg.scalar_mult(gamma[i], g1) for i in range(total_dim)],
            'params': self.bg.params
        }

        # Generate encryption keys for each slot
        self.ek = {}
        for i in range(n_slots):
            self.ek[i] = {
                'alpha': self.bg.random_element("Zr"),
                'beta': self.bg.random_element("Zr"),
                'gamma': self.bg.random_element("Zr")
            }

        # Master secret key
        self.msk = {
            'alpha': alpha,
            'beta': beta,
            'gamma': gamma
        }

        self.current_iteration += 1
        return self.pp, self.ek, self.msk

    def encrypt(self, slot_idx: int, data: np.ndarray) -> List:
        """
        Encryption algorithm for client data

        Args:
            slot_idx: Encryption slot index (client ID)
            data: Input data vector (flattened)

        Returns:
            Ciphertext
        """
        if slot_idx not in self.ek:
            raise ValueError(f"No encryption key for slot {slot_idx}")

        ek_i = self.ek[slot_idx]

        # Ensure data is flattened
        if data.ndim > 1:
            data = data.flatten()

        dim = len(data)

        # Encryption process
        ct_elements = []
        for j in range(dim):
            # Convert to scalar for encryption
            data_value = float(data[j])

            # Compute encryption elements
            exponent_alpha = (ek_i['alpha'] * int(data_value * 1000)) % self.order
            term_alpha = self.bg.scalar_mult(exponent_alpha, self.pp['h1_i'][slot_idx])

            exponent_beta = (ek_i['beta'] * int(data_value * 1000)) % self.order
            term_beta = self.bg.scalar_mult(exponent_beta, self.pp['h2_i'][slot_idx])

            # Add randomness
            r = self.bg.random_element("Zr")
            random_alpha = self.bg.scalar_mult(r, self.pp['g1'])
            random_beta = self.bg.scalar_mult(r, self.pp['g2'])

            ct_element_alpha = self.bg.add_points(term_alpha, random_alpha)
            ct_element_beta = self.bg.add_points(term_beta, random_beta)

            ct_elements.append((ct_element_alpha, ct_element_beta))

        return ct_elements

    def keygen(self, c: np.ndarray) -> List:
        """
        Key generation for function vector c

        Args:
            c: Function vector

        Returns:
            Decryption key
        """
        if self.msk is None:
            raise ValueError("Setup must be called before key generation")

        dim = len(c)
        sk_elements = []

        # Ensure we don't exceed the length of msk elements
        min_dim = min(dim, len(self.msk['alpha']), len(self.msk['beta']))

        for i in range(min_dim):
            # Compute key element
            exponent = (c[i] * self.msk['alpha'][i] * self.msk['beta'][i]) % self.order
            sk_element = self.bg.scalar_mult(exponent, self.pp['g1'])
            sk_elements.append(sk_element)

        return sk_elements

    def decrypt(self, ciphertexts: List, sk: List) -> float:
        """
        Decryption algorithm

        Args:
            ciphertexts: List of ciphertexts from all slots
            sk: Decryption key

        Returns:
            Decrypted function value
        """
        # Reset pairing counter before decryption
        self.bg.reset_pairing_count()

        # Compute pairing product
        result = self.bg.GT['one']

        # Simplified decryption for quadratic function
        # In a real implementation, this would follow the full qMIFE decryption algorithm
        for i, ct in enumerate(ciphertexts):
            for j, (ct_alpha, ct_beta) in enumerate(ct):
                if i < len(sk) and j < len(sk):
                    pairing_result = self.bg.pairing(ct_alpha, ct_beta)

                    # Handle different types of sk_element
                    sk_element = sk[i]
                    if isinstance(sk_element, tuple):
                        sk_value = sk_element[0] if sk_element[0] is not None else 1
                    else:
                        sk_value = sk_element

                    # Convert to integer if needed
                    try:
                        sk_value = int(sk_value)
                    except (ValueError, TypeError):
                        sk_value = 1

                    # Apply key element
                    sk_power = pow(pairing_result, sk_value, self.bg.params.p)
                    result = (result * sk_power) % self.bg.params.p

        # Map from GT group element to real number
        return self._map_from_gt(result)

    def _map_from_gt(self, gt_element) -> float:
        """
        Map from GT group element to real number
        Note: This is a simplified implementation for research purposes
        """
        # Hash group element and map to [-1, 1] range
        element_hash = hashlib.sha256(str(gt_element).encode()).digest()
        int_value = int.from_bytes(element_hash[:8], 'big') / (2 ** 64 - 1)
        return float(int_value * 2 - 1)  # Map to [-1, 1]

    def get_pairing_count(self):
        """Get the current pairing count"""
        return self.bg.pairing_count


class SFedV:
    """
    SFedV: Secure Vertical Federated Learning using Quadratic MIFE
    Based on: Chen et al. "Quadratic Functional Encryption for Secure Training in Vertical Federated Learning"
    """

    def __init__(self, n_clients: int, feature_dims: List[int], security_param: int = 128):
        """
        Initialize SFedV protocol

        Args:
            n_clients: Number of clients
            feature_dims: List of feature dimensions for each client
            security_param: Security parameter
        """
        self.n_clients = n_clients
        self.feature_dims = feature_dims
        self.total_features = sum(feature_dims)
        self.security_param = security_param

        # Initialize qMIFE
        self.qmife = QuadraticMIFE(security_param)

        # Model weights (held by aggregator)
        self.weights = None
        self.label_client_idx = 0  # Client that holds labels

        # Performance statistics
        self.stats = {
            "setup_time": 0,
            "encrypt_time": 0,
            "keygen_time": 0,
            "decrypt_time": 0,
            "pairing_count": 0
        }

    def initialize_weights(self):
        """Initialize model weights"""
        self.weights = []
        for dim in self.feature_dims:
            self.weights.append(np.random.randn(dim))

    def setup_iteration(self):
        """Setup for a new training iteration (new qMIFE instance)"""
        start_time = time.time()
        self.pp, self.ek, self.msk = self.qmife.setup(self.n_clients + 1)  # +1 for labels
        self.stats["setup_time"] += time.time() - start_time

    def construct_function_vectors(self, batch_size: int):
        """
        Construct function vectors for gradient computation
        Based on paper's approach for constructing c vectors

        Args:
            batch_size: Size of training batch

        Returns:
            List of function vectors for gradient computation
        """
        function_vectors = []

        # For each client and each feature, construct a function vector
        for client_idx in range(self.n_clients):
            for feature_idx in range(self.feature_dims[client_idx]):
                # The function vector should have the same length as the msk elements
                # In our setup, msk elements have length n_clients + 1
                c_vector = np.zeros(self.n_clients + 1)

                # Set the appropriate coefficient for this client
                if client_idx < len(c_vector):
                    c_vector[client_idx] = 1.0

                function_vectors.append((client_idx, feature_idx, c_vector))

        return function_vectors

    def client_encrypt(self, client_idx: int, data: np.ndarray, labels: np.ndarray = None):
        """
        Client-side encryption

        Args:
            client_idx: Client identifier
            data: Feature data (2D array)
            labels: Labels (1D array, if this client holds them)

        Returns:
            Encrypted data
        """
        start_time = time.time()

        # Flatten the data for encryption
        flattened_data = data.flatten()

        # Encrypt features
        encrypted_data = self.qmife.encrypt(client_idx, flattened_data)

        # Encrypt labels if this client holds them
        encrypted_labels = None
        if labels is not None and client_idx == self.label_client_idx:
            # Ensure labels are 1D array
            if labels.ndim > 1:
                labels = labels.flatten()
            encrypted_labels = self.qmife.encrypt(self.n_clients, labels)  # Use special slot for labels

        self.stats["encrypt_time"] += time.time() - start_time

        return encrypted_data, encrypted_labels

    def compute_gradient(self, encrypted_data: List, encrypted_labels: List, function_vectors: List):
        """
        Compute gradient from encrypted data

        Args:
            encrypted_data: List of encrypted data from all clients
            encrypted_labels: Encrypted labels
            function_vectors: List of function vectors for gradient computation

        Returns:
            Gradient vector
        """
        start_time = time.time()
        gradient = np.zeros(self.total_features)

        # Combine all ciphertexts (data + labels)
        all_ciphertexts = encrypted_data.copy()
        if encrypted_labels is not None:
            all_ciphertexts.append(encrypted_labels)

        # For each function vector, compute the corresponding gradient component
        for client_idx, feature_idx, c_vector in function_vectors:
            # Generate decryption key for this function vector
            keygen_start = time.time()
            sk = self.qmife.keygen(c_vector)
            self.stats["keygen_time"] += time.time() - keygen_start

            # Decrypt to get gradient component
            decrypt_start = time.time()
            grad_component = self.qmife.decrypt(all_ciphertexts, sk)
            self.stats["decrypt_time"] += time.time() - decrypt_start
            self.stats["pairing_count"] += self.qmife.get_pairing_count()

            # Store in gradient vector
            pos = sum(self.feature_dims[:client_idx]) + feature_idx
            if pos < len(gradient):
                gradient[pos] = grad_component

        return gradient

    def update_weights(self, gradient: np.ndarray, learning_rate: float = 0.01):
        """
        Update model weights

        Args:
            gradient: Computed gradient
            learning_rate: Learning rate for weight update
        """
        if self.weights is None:
            raise ValueError("Weights not initialized")

        # Convert flattened gradient back to per-client weights
        start_idx = 0
        for i, dim in enumerate(self.feature_dims):
            end_idx = start_idx + dim
            if end_idx <= len(gradient):
                self.weights[i] -= learning_rate * gradient[start_idx:end_idx]
            start_idx = end_idx

    def train_iteration(self, client_data: List, labels: np.ndarray, batch_size: int, learning_rate: float = 0.01):
        """
        Complete training iteration

        Args:
            client_data: List of data from each client
            labels: Training labels
            batch_size: Size of training batch
            learning_rate: Learning rate for weight update

        Returns:
            Computed gradient
        """
        # Setup new qMIFE instance for this iteration
        self.setup_iteration()

        # Clients encrypt their data
        encrypted_data = []
        encrypted_labels = None

        for i, data in enumerate(client_data):
            if i == self.label_client_idx:
                enc_data, enc_labels = self.client_encrypt(i, data, labels)
                encrypted_data.append(enc_data)
                encrypted_labels = enc_labels
            else:
                enc_data, _ = self.client_encrypt(i, data)
                encrypted_data.append(enc_data)

        # Construct function vectors for gradient computation
        function_vectors = self.construct_function_vectors(batch_size)

        # Compute gradient
        gradient = self.compute_gradient(encrypted_data, encrypted_labels, function_vectors)

        # Update weights
        self.update_weights(gradient, learning_rate)

        return gradient

    def get_performance_stats(self):
        """Get performance statistics"""
        return self.stats.copy()

    def reset_stats(self):
        """Reset performance statistics"""
        self.stats = {
            "setup_time": 0,
            "encrypt_time": 0,
            "keygen_time": 0,
            "decrypt_time": 0,
            "pairing_count": 0
        }


class SFedVEvaluator:
    """
    SFedV Evaluator - For performance testing and analysis
    """

    def __init__(self):
        self.results = {}

    def evaluate_performance(self, n_clients_list: List[int], feature_dims_list: List[List[int]],
                             batch_sizes: List[int] = [32], security_params: List[int] = [128],
                             n_trials: int = 3) -> Dict:
        """
        Comprehensive SFedV performance evaluation
        """
        results = {}

        for security_param in security_params:
            for n_clients in n_clients_list:
                for feature_dims in feature_dims_list:
                    # Skip mismatched configurations
                    if n_clients != len(feature_dims):
                        print(f"Skipping mismatched config: {n_clients} clients vs {len(feature_dims)} feature dims")
                        continue

                    for batch_size in batch_sizes:
                        key = f"sec_{security_param}_clients_{n_clients}_dims_{'-'.join(map(str, feature_dims))}_batch_{batch_size}"
                        results[key] = {
                            "setup_times": [],
                            "encrypt_times": [],
                            "keygen_times": [],
                            "decrypt_times": [],
                            "pairing_counts": [],
                            "avg_setup_time": 0,
                            "avg_encrypt_time": 0,
                            "avg_keygen_time": 0,
                            "avg_decrypt_time": 0,
                            "avg_pairing_count": 0
                        }

                        for trial in range(n_trials):
                            try:
                                # Create SFedV instance
                                sfedv = SFedV(n_clients, feature_dims, security_param)
                                sfedv.initialize_weights()

                                # Generate test data
                                client_data = []
                                for dim in feature_dims:
                                    client_data.append(np.random.randn(batch_size, dim))

                                labels = np.random.randn(batch_size)

                                # Run training iteration
                                gradient = sfedv.train_iteration(client_data, labels, batch_size)

                                # Record performance
                                stats = sfedv.get_performance_stats()
                                results[key]["setup_times"].append(stats["setup_time"])
                                results[key]["encrypt_times"].append(stats["encrypt_time"])
                                results[key]["keygen_times"].append(stats["keygen_time"])
                                results[key]["decrypt_times"].append(stats["decrypt_time"])
                                results[key]["pairing_counts"].append(stats["pairing_count"])

                                print(
                                    f"Trial {trial + 1}/{n_trials} completed: {key}, gradient norm: {np.linalg.norm(gradient):.6f}")

                            except Exception as e:
                                print(f"Trial {trial + 1} failed: {e}")
                                continue

                        # Calculate averages
                        for metric in ["setup_times", "encrypt_times", "keygen_times", "decrypt_times",
                                       "pairing_counts"]:
                            if results[key][metric]:
                                results[key][
                                    f"avg_{metric.split('_')[0]}_time" if "time" in metric else f"avg_{metric}"] = np.mean(
                                    results[key][metric])

        self.results['performance'] = results
        return results

    def plot_performance(self, save_path: str = None) -> None:
        """
        Plot performance charts
        """
        if 'performance' not in self.results:
            print("No performance data available")
            return

        results = self.results['performance']

        # Prepare labels
        setup_title = 'Average Setup Time'
        encrypt_title = 'Average Encryption Time'
        decrypt_title = 'Average Decryption Time'
        pairing_title = 'Average Pairing Count'
        time_label = 'Time (seconds)'
        count_label = 'Count'

        # Prepare data
        keys = list(results.keys())
        setup_times = [results[key]['avg_setup_time'] for key in keys]
        encrypt_times = [results[key]['avg_encrypt_time'] for key in keys]
        keygen_times = [results[key]['avg_keygen_time'] for key in keys]
        decrypt_times = [results[key]['avg_decrypt_time'] for key in keys]
        pairing_counts = [results[key]['avg_pairing_count'] for key in keys]

        # Create charts
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # Setup time
        ax1.bar(keys, setup_times)
        ax1.set_title(setup_title)
        ax1.set_ylabel(time_label)
        ax1.tick_params(axis='x', rotation=45)

        # Encryption time
        ax2.bar(keys, encrypt_times)
        ax2.set_title(encrypt_title)
        ax2.set_ylabel(time_label)
        ax2.tick_params(axis='x', rotation=45)

        # Decryption time
        ax3.bar(keys, decrypt_times)
        ax3.set_title(decrypt_title)
        ax3.set_ylabel(time_label)
        ax3.tick_params(axis='x', rotation=45)

        # Pairing count
        ax4.bar(keys, pairing_counts)
        ax4.set_title(pairing_title)
        ax4.set_ylabel(count_label)
        ax4.tick_params(axis='x', rotation=45)

        plt.tight_layout()

        if save_path:
            plt.savefig(f"{save_path}_performance.png", dpi=300, bbox_inches='tight')
        plt.show()

    def save_results(self, filepath: str) -> None:
        """Save evaluation results"""
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)

    def load_results(self, filepath: str) -> None:
        """Load evaluation results"""
        with open(filepath, 'r') as f:
            self.results = json.load(f)


# Example usage for logistic regression (using Taylor approximation)
class LogisticSFedV(SFedV):
    """
    SFedV extension for logistic regression using Taylor approximation
    """

    def __init__(self, n_clients: int, feature_dims: List[int], security_param: int = 128):
        super().__init__(n_clients, feature_dims, security_param)

    def train_iteration(self, client_data: List, labels: np.ndarray, batch_size: int, learning_rate: float = 0.01):
        """
        Training iteration for logistic regression
        Uses Taylor approximation as described in the paper
        """
        # Adjust labels for logistic regression (convert to {-0.5, 0.5})
        adjusted_labels = labels - 0.5

        # Adjust weights for Taylor approximation (multiply by 1/4)
        for i in range(len(self.weights)):
            self.weights[i] = self.weights[i] / 4

        # Run standard training iteration
        gradient = super().train_iteration(client_data, adjusted_labels, batch_size, learning_rate)

        # Restore original weight scaling
        for i in range(len(self.weights)):
            self.weights[i] = self.weights[i] * 4

        return gradient


# Demo function
def demo():
    """Demonstrate SFedV protocol"""
    print("SFedV: Secure Vertical Federated Learning Demo")
    print("=" * 50)

    # Create SFedV instance
    n_clients = 2
    feature_dims = [2, 3]  # Client 0 has 2 features, Client 1 has 3 features
    sfedv = SFedV(n_clients, feature_dims)
    sfedv.initialize_weights()

    print(f"Initialization completed: {n_clients} clients, feature dimensions {feature_dims}")
    print(f"Weights initialized: {[w.shape for w in sfedv.weights]}")

    # Generate test data
    batch_size = 10
    client_data = [
        np.random.randn(batch_size, feature_dims[0]),  # Client 0 data
        np.random.randn(batch_size, feature_dims[1])  # Client 1 data
    ]
    labels = np.random.randn(batch_size)  # Labels held by client 0

    # Run training iteration
    gradient = sfedv.train_iteration(client_data, labels, batch_size)

    print(f"Training iteration completed")
    print(f"Gradient norm: {np.linalg.norm(gradient):.6f}")

    # Show performance statistics
    stats = sfedv.get_performance_stats()
    print(f"Setup time: {stats['setup_time']:.4f} seconds")
    print(f"Encryption time: {stats['encrypt_time']:.4f} seconds")
    print(f"Key generation time: {stats['keygen_time']:.4f} seconds")
    print(f"Decryption time: {stats['decrypt_time']:.4f} seconds")
    print(f"Pairing operations: {stats['pairing_count']}")

    # Performance evaluation
    print("\nPerformance Evaluation")
    print("=" * 50)

    evaluator = SFedVEvaluator()
    results = evaluator.evaluate_performance(
        n_clients_list=[2, 3],
        feature_dims_list=[[2, 2], [3, 3, 3]],
        batch_sizes=[10, 20],
        security_params=[128],
        n_trials=2
    )

    # Plot performance charts
    evaluator.plot_performance("sfedv_performance")

    print("Demo completed!")


if __name__ == "__main__":
    demo()