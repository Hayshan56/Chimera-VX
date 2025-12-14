#!/usr/bin/env python3
# chimera-vx/puzzles/01_quantum/verifier.py
# Quantum puzzle verifier for Circle 1

import json
import hashlib
import secrets
import random
import math
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import logging
from datetime import datetime
import base64
import qiskit
from qiskit import QuantumCircuit, Aer, execute
from qiskit.quantum_info import Statevector, partial_trace, entropy
import re

logger = logging.getLogger(__name__)

class QuantumPuzzleVerifier:
    """Verify solutions for quantum puzzles"""
    
    def __init__(self, config_path: str = "templates/quantum_template.json"):
        self.config = self.load_config(config_path)
        self.thresholds = {
            'probability_tolerance': 0.01,
            'entropy_tolerance': 0.1,
            'min_shots': 1000,
            'max_verification_time': 30  # seconds
        }
    
    def load_config(self, config_path: str) -> Dict:
        """Load quantum puzzle configuration"""
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def verify_solution(self, puzzle_data: Dict, submitted_solution: str,
                       player_id: int) -> Tuple[bool, Dict]:
        """Verify a submitted solution"""
        
        logger.info(f"Verifying quantum solution for player {player_id}")
        
        try:
            # Extract required information
            circuit_qasm = puzzle_data['files']['circuit.qasm']
            expected_solution_hash = puzzle_data.get('expected_hash')
            
            # Parse submitted solution
            parsed_solution = self.parse_solution(submitted_solution)
            if not parsed_solution:
                return False, {'error': 'Invalid solution format'}
            
            # Simulate circuit
            simulation_result = self.simulate_circuit(circuit_qasm)
            if not simulation_result['success']:
                return False, {'error': 'Simulation failed'}
            
            # Calculate expected solution from simulation
            expected_solution = self.calculate_expected_solution(simulation_result)
            
            # Compare solutions
            verification_result = self.compare_solutions(
                submitted=parsed_solution,
                expected=expected_solution,
                simulation=simulation_result
            )
            
            # Check for cheating patterns
            cheat_analysis = self.analyze_for_cheating(
                submitted_solution=submitted_solution,
                simulation_result=simulation_result,
                player_id=player_id
            )
            
            # Prepare verification data
            verification_data = {
                'verification_success': verification_result['match'],
                'simulation_data': simulation_result,
                'solution_comparison': verification_result,
                'cheat_analysis': cheat_analysis,
                'timestamp': datetime.now().isoformat(),
                'verification_method': 'quantum_simulation_and_analysis'
            }
            
            if verification_result['match']:
                logger.info(f"Solution verified successfully for player {player_id}")
                return True, verification_data
            else:
                logger.info(f"Solution verification failed for player {player_id}")
                return False, verification_data
            
        except Exception as e:
            logger.error(f"Error in solution verification: {e}")
            return False, {'error': str(e)}
    
    def parse_solution(self, solution_str: str) -> Optional[Dict]:
        """Parse submitted solution string"""
        # Expected format: STATE_PROBABILITY_ENTROPY_HASH
        pattern = r'^([01]+)_([0-9.]+)_([0-9.]+)_([0-9a-f]+)$'
        
        match = re.match(pattern, solution_str)
        if not match:
            return None
        
        return {
            'state': match.group(1),
            'probability': float(match.group(2)),
            'entropy': float(match.group(3)),
            'hash': match.group(4)
        }
    
    def simulate_circuit(self, qasm_str: str) -> Dict:
        """Simulate quantum circuit"""
        try:
            # Load circuit
            circuit = QuantumCircuit.from_qasm_str(qasm_str)
            
            # Run simulation with multiple backends
            results = {}
            
            # 1. QASM simulator (for measurement statistics)
            backend_qasm = Aer.get_backend('qasm_simulator')
            result_qasm = execute(circuit, backend_qasm, shots=8192).result()
            counts = result_qasm.get_counts()
            
            # Calculate statistics
            total_shots = sum(counts.values())
            probabilities = {state: count/total_shots for state, count in counts.items()}
            most_frequent = max(counts, key=counts.get)
            most_freq_prob = probabilities[most_frequent]
            
            # Calculate entropy
            shannon_entropy = 0.0
            for prob in probabilities.values():
                if prob > 0:
                    shannon_entropy -= prob * math.log2(prob)
            
            # 2. Statevector simulator (for small circuits)
            statevector_data = None
            if circuit.num_qubits <= 8:
                try:
                    backend_sv = Aer.get_backend('statevector_simulator')
                    circuit_no_measure = circuit.remove_final_measurements(inplace=False)
                    result_sv = execute(circuit_no_measure, backend_sv).result()
                    statevector = result_sv.get_statevector()
                    
                    # Calculate entanglement entropy
                    if circuit.num_qubits >= 2:
                        # Trace out half the qubits
                        subsystem_a = list(range(circuit.num_qubits // 2))
                        subsystem_b = list(range(circuit.num_qubits // 2, circuit.num_qubits))
                        
                        # Calculate reduced density matrices
                        rho_a = partial_trace(statevector, subsystem_b)
                        rho_b = partial_trace(statevector, subsystem_a)
                        
                        # Calculate von Neumann entropy
                        entropy_a = entropy(rho_a)
                        entropy_b = entropy(rho_b)
                        
                        statevector_data = {
                            'statevector': statevector.data.tolist(),
                            'entanglement_entropy_a': float(entropy_a),
                            'entanglement_entropy_b': float(entropy_b)
                        }
                except:
                    pass
            
            return {
                'success': True,
                'circuit_info': {
                    'qubits': circuit.num_qubits,
                    'depth': circuit.depth(),
                    'gates': len(circuit.data)
                },
                'measurement': {
                    'counts': counts,
                    'probabilities': probabilities,
                    'most_frequent_state': most_frequent,
                    'most_frequent_probability': most_freq_prob,
                    'total_shots': total_shots,
                    'unique_states': len(counts),
                    'shannon_entropy': shannon_entropy
                },
                'statevector': statevector_data,
                'simulation_time': 0.1  # Placeholder
            }
            
        except Exception as e:
            logger.error(f"Circuit simulation error: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def calculate_expected_solution(self, simulation_result: Dict) -> Dict:
        """Calculate expected solution from simulation"""
        
        measurement = simulation_result['measurement']
        
        # Extract most frequent state and probability
        state = measurement['most_frequent_state']
        probability = measurement['most_frequent_probability']
        entropy_val = measurement['shannon_entropy']
        
        # Create a hash from the simulation data
        hash_data = f"{state}_{probability:.6f}_{entropy_val:.4f}"
        solution_hash = hashlib.sha256(hash_data.encode()).hexdigest()[:16]
        
        return {
            'state': state,
            'probability': probability,
            'entropy': entropy_val,
            'hash': solution_hash
        }
    
    def compare_solutions(self, submitted: Dict, expected: Dict, 
                         simulation: Dict) -> Dict:
        """Compare submitted and expected solutions"""
        
        # Check state match
        state_match = submitted['state'] == expected['state']
        
        # Check probability (within tolerance)
        prob_diff = abs(submitted['probability'] - expected['probability'])
        prob_match = prob_diff <= self.thresholds['probability_tolerance']
        
        # Check entropy (within tolerance)
        entropy_diff = abs(submitted['entropy'] - expected['entropy'])
        entropy_match = entropy_diff <= self.thresholds['entropy_tolerance']
        
        # Check hash (exact match required)
        hash_match = submitted['hash'] == expected['hash']
        
        # Calculate overall match score
        score = 0
        if state_match: score += 0.4
        if prob_match: score += 0.3
        if entropy_match: score += 0.2
        if hash_match: score += 0.1
        
        # Determine if solution is correct
        is_correct = score >= 0.95  # 95% match required
        
        return {
            'match': is_correct,
            'score': score,
            'components': {
                'state_match': {'match': state_match, 'value': submitted['state']},
                'probability_match': {
                    'match': prob_match,
                    'submitted': submitted['probability'],
                    'expected': expected['probability'],
                    'difference': prob_diff
                },
                'entropy_match': {
                    'match': entropy_match,
                    'submitted': submitted['entropy'],
                    'expected': expected['entropy'],
                    'difference': entropy_diff
                },
                'hash_match': {'match': hash_match, 'value': submitted['hash']}
            },
            'thresholds': self.thresholds
        }
    
    def analyze_for_cheating(self, submitted_solution: str, 
                           simulation_result: Dict, player_id: int) -> Dict:
        """Analyze solution for cheating patterns"""
        
        cheat_flags = []
        
        # 1. Check if solution is too perfect
        if self.is_too_perfect(submitted_solution, simulation_result):
            cheat_flags.append('solution_too_perfect')
        
        # 2. Check for common cheat patterns
        if self.contains_cheat_patterns(submitted_solution):
            cheat_flags.append('contains_cheat_patterns')
        
        # 3. Check timing (if available)
        # This would check if solution was submitted too quickly
        
        # 4. Check for statistical anomalies
        if self.has_statistical_anomalies(simulation_result):
            cheat_flags.append('statistical_anomalies')
        
        # 5. Check solution format
        if not self.is_valid_solution_format(submitted_solution):
            cheat_flags.append('invalid_format')
        
        return {
            'cheat_detected': len(cheat_flags) > 0,
            'flags': cheat_flags,
            'suspicion_score': len(cheat_flags) * 20,
            'player_id': player_id
        }
    
    def is_too_perfect(self, solution: str, simulation: Dict) -> bool:
        """Check if solution is suspiciously perfect"""
        # A perfect solution would have exactly matching probabilities
        # down to many decimal places
        
        # Extract probability from solution
        match = re.search(r'_([0-9.]+)_', solution)
        if not match:
            return False
        
        submitted_prob = float(match.group(1))
        expected_prob = simulation['measurement']['most_frequent_probability']
        
        # Check if probability matches exactly (too many decimal places)
        prob_str = match.group(1)
        decimal_places = len(prob_str.split('.')[1]) if '.' in prob_str else 0
        
        # More than 10 decimal places is suspicious
        if decimal_places > 10:
            return True
        
        # Exact match to many decimal places is suspicious
        if abs(submitted_prob - expected_prob) < 1e-12:
            return True
        
        return False
    
    def contains_cheat_patterns(self, solution: str) -> bool:
        """Check for known cheat patterns"""
        
        cheat_patterns = [
            r'^11111111_',  # All ones pattern
            r'^00000000_',  # All zeros pattern
            r'FLAG{.*}',    # Flag pattern
            r'CTF{.*}',     # CTF pattern
            r'CHEAT',       # Literal cheat
            r'ADMIN',       # Admin references
            r'BACKDOOR',    # Backdoor references
        ]
        
        for pattern in cheat_patterns:
            if re.search(pattern, solution, re.IGNORECASE):
                return True
        
        return False
    
    def has_statistical_anomalies(self, simulation: Dict) -> bool:
        """Check for statistical anomalies in simulation"""
        
        measurement = simulation['measurement']
        
        # Check if distribution is too uniform (possible random generation)
        probabilities = list(measurement['probabilities'].values())
        if len(probabilities) > 1:
            variance = np.var(probabilities)
            
            # Very low variance might indicate fabricated results
            if variance < 1e-10:
                return True
        
        # Check if most frequent state has suspicious probability
        max_prob = measurement['most_frequent_probability']
        
        # Probability of 1.0 is impossible for multi-qubit circuits
        if abs(max_prob - 1.0) < 1e-10 and measurement['unique_states'] > 1:
            return True
        
        # Probability too close to simple fractions might be suspicious
        simple_fractions = [0.5, 0.25, 0.125, 0.0625, 0.333, 0.667]
        for fraction in simple_fractions:
            if abs(max_prob - fraction) < 0.001:
                return True
        
        return False
    
    def is_valid_solution_format(self, solution: str) -> bool:
        """Check if solution has valid format"""
        pattern = r'^[01]+_[0-9.]+_[0-9.]+_[0-9a-f]+$'
        return bool(re.match(pattern, solution))
    
    def generate_hints(self, verification_result: Dict, 
                      puzzle_data: Dict) -> List[str]:
        """Generate hints based on verification result"""
        
        hints = []
        
        if not verification_result['match']:
            comparison = verification_result['components']
            
            if not comparison['state_match']['match']:
                hints.append(f"Your state doesn't match. Check your measurements.")
            
            if not comparison['probability_match']['match']:
                diff = comparison['probability_match']['difference']
                hints.append(f"Probability is off by {diff:.6f}. Run more simulation shots.")
            
            if not comparison['entropy_match']['match']:
                hints.append("Entropy value is incorrect. Check your calculations.")
            
            if not comparison['hash_match']['match']:
                hints.append("Solution hash doesn't match. Verify all components.")
        
        # Add puzzle-specific hints
        hints.extend(self.config['hints'][:2])
        
        return hints
    
    def verify_offline(self, puzzle_dir: str, submitted_solution: str) -> Tuple[bool, Dict]:
        """Verify solution offline (for testing)"""
        
        try:
            # Load puzzle data
            with open(f"{puzzle_dir}/metadata.json", 'r') as f:
                metadata = json.load(f)
            
            # Load circuit QASM
            with open(f"{puzzle_dir}/circuit.qasm", 'r') as f:
                qasm_str = f.read()
            
            # Load circuit info
            with open(f"{puzzle_dir}/circuit_info.json", 'r') as f:
                circuit_info = json.load(f)
            
            # Create puzzle data
            puzzle_data = {
                'files': {
                    'circuit.qasm': qasm_str,
                    'circuit_info.json': json.dumps(circuit_info)
                }
            }
            
            # Verify solution
            return self.verify_solution(
                puzzle_data=puzzle_data,
                submitted_solution=submitted_solution,
                player_id=metadata['player_id']
            )
            
        except Exception as e:
            logger.error(f"Offline verification error: {e}")
            return False, {'error': str(e)}


# Test function
def test_verifier():
    """Test the quantum puzzle verifier"""
    print("Testing Quantum Puzzle Verifier...")
    
    # Create a simple test circuit
    circuit = QuantumCircuit(2, 2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure([0, 1], [0, 1])
    
    qasm_str = circuit.qasm()
    
    # Create puzzle data
    puzzle_data = {
        'files': {
            'circuit.qasm': qasm_str,
            'circuit_info.json': json.dumps({
                'qubits': 2,
                'gates': 3,
                'depth': 2
            })
        }
    }
    
    # Create verifier
    verifier = QuantumPuzzleVerifier()
    
    # Simulate to get expected solution
    simulation = verifier.simulate_circuit(qasm_str)
    expected = verifier.calculate_expected_solution(simulation)
    
    # Create correct solution
    correct_solution = f"{expected['state']}_{expected['probability']:.6f}_{expected['entropy']:.4f}_{expected['hash']}"
    
    # Test correct solution
    print("\\n1. Testing CORRECT solution:")
    success, data = verifier.verify_solution(
        puzzle_data=puzzle_data,
        submitted_solution=correct_solution,
        player_id=1
    )
    print(f"Success: {success}")
    print(f"Score: {data.get('solution_comparison', {}).get('score', 0)}")
    
    # Test incorrect solution
    print("\\n2. Testing INCORRECT solution:")
    wrong_solution = "00_0.250000_1.0000_abcdef1234567890"
    success, data = verifier.verify_solution(
        puzzle_data=puzzle_data,
        submitted_solution=wrong_solution,
        player_id=1
    )
    print(f"Success: {success}")
    print(f"Score: {data.get('solution_comparison', {}).get('score', 0)}")
    
    # Test cheat detection
    print("\\n3. Testing CHEAT detection:")
    cheat_solution = "FLAG{THIS_IS_CHEATING}"
    success, data = verifier.verify_solution(
        puzzle_data=puzzle_data,
        submitted_solution=cheat_solution,
        player_id=1
    )
    print(f"Success: {success}")
    print(f"Cheat detected: {data.get('cheat_analysis', {}).get('cheat_detected', False)}")
    
    return verifier


if __name__ == "__main__":
    test_verifier()
