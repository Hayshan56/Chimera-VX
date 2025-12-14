#!/usr/bin/env python3
# chimera-vx/puzzles/01_quantum/generator.py
# Quantum puzzle generator for Circle 1

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
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter
from qiskit.circuit.library import (RXGate, RYGate, RZGate, CXGate, CZGate, 
                                   SwapGate, CCXGate, HGate, XGate, YGate, ZGate)

logger = logging.getLogger(__name__)

class QuantumPuzzleGenerator:
    """Generate quantum computing puzzles for Chimera-VX"""
    
    def __init__(self, config_path: str = "templates/quantum_template.json"):
        self.config = self.load_config(config_path)
        self.gate_library = self.initialize_gate_library()
        
    def load_config(self, config_path: str) -> Dict:
        """Load quantum puzzle configuration"""
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def initialize_gate_library(self) -> Dict:
        """Initialize quantum gate library"""
        return {
            'h': {'gate': HGate, 'params': 0, 'description': 'Hadamard gate'},
            'x': {'gate': XGate, 'params': 0, 'description': 'Pauli-X gate'},
            'y': {'gate': YGate, 'params': 0, 'description': 'Pauli-Y gate'},
            'z': {'gate': ZGate, 'params': 0, 'description': 'Pauli-Z gate'},
            's': {'gate': lambda: QuantumCircuit(1).s(0), 'params': 0, 'description': 'Phase gate'},
            't': {'gate': lambda: QuantumCircuit(1).t(0), 'params': 0, 'description': 'T gate'},
            'rx': {'gate': RXGate, 'params': 1, 'description': 'Rotation around X axis'},
            'ry': {'gate': RYGate, 'params': 1, 'description': 'Rotation around Y axis'},
            'rz': {'gate': RZGate, 'params': 1, 'description': 'Rotation around Z axis'},
            'cx': {'gate': CXGate, 'params': 0, 'description': 'Controlled-NOT gate'},
            'cz': {'gate': CZGate, 'params': 0, 'description': 'Controlled-Z gate'},
            'swap': {'gate': SwapGate, 'params': 0, 'description': 'SWAP gate'},
            'ccx': {'gate': CCXGate, 'params': 0, 'description': 'Toffoli gate'}
        }
    
    def generate_puzzle(self, player_id: int, circle_number: int, 
                       difficulty_multiplier: float = 1.0) -> Dict:
        """Generate a unique quantum puzzle for a player"""
        
        logger.info(f"Generating quantum puzzle for player {player_id}")
        
        # Set random seed based on player ID and timestamp
        seed = f"{player_id}_{circle_number}_{int(datetime.now().timestamp())}"
        random.seed(int(hashlib.sha256(seed.encode()).hexdigest(), 16) % 2**32)
        np.random.seed(int(hashlib.sha256(seed.encode()).hexdigest(), 16) % 2**32)
        
        # Determine puzzle parameters based on circle
        num_qubits = self.determine_qubits(circle_number, difficulty_multiplier)
        num_gates = self.determine_gates(circle_number, difficulty_multiplier)
        
        # Generate quantum circuit
        circuit, circuit_info = self.generate_circuit(num_qubits, num_gates, player_id)
        
        # Generate hidden message
        hidden_message = self.generate_hidden_message(player_id, circle_number)
        
        # Encode message in quantum state
        encoded_circuit = self.encode_message_in_circuit(circuit, hidden_message)
        
        # Calculate expected solution
        solution = self.calculate_solution(encoded_circuit, hidden_message)
        
        # Generate puzzle data
        puzzle_data = self.create_puzzle_data(
            circuit=encoded_circuit,
            circuit_info=circuit_info,
            player_id=player_id,
            circle_number=circle_number,
            hidden_message=hidden_message
        )
        
        # Generate solution hash
        solution_hash = hashlib.sha256(solution.encode()).hexdigest()
        
        logger.info(f"Generated quantum puzzle with {num_qubits} qubits and {num_gates} gates")
        
        return {
            'puzzle_data': puzzle_data,
            'solution': solution,
            'solution_hash': solution_hash,
            'metadata': {
                'player_id': player_id,
                'circle': circle_number,
                'qubits': num_qubits,
                'gates': num_gates,
                'generated_at': datetime.now().isoformat(),
                'puzzle_id': f"quantum_{player_id}_{circle_number}_{secrets.token_hex(4)}"
            }
        }
    
    def determine_qubits(self, circle: int, difficulty: float) -> int:
        """Determine number of qubits based on circle"""
        min_qubits = self.config['circuit_parameters']['min_qubits']
        max_qubits = self.config['circuit_parameters']['max_qubits']
        
        # Scale qubits with circle
        base_qubits = min_qubits + (circle - 1) * 2
        qubits = min(base_qubits, max_qubits)
        
        # Apply difficulty multiplier
        qubits = int(qubits * difficulty)
        
        return max(min_qubits, min(qubits, max_qubits))
    
    def determine_gates(self, circle: int, difficulty: float) -> int:
        """Determine number of gates based on circle"""
        min_gates = self.config['circuit_parameters']['min_gates']
        max_gates = self.config['circuit_parameters']['max_gates']
        
        # Scale gates with circle
        base_gates = min_gates + (circle - 1) * 10
        gates = min(base_gates, max_gates)
        
        # Apply difficulty multiplier
        gates = int(gates * difficulty)
        
        return max(min_gates, min(gates, max_gates))
    
    def generate_circuit(self, num_qubits: int, num_gates: int, 
                        player_id: int) -> Tuple[QuantumCircuit, Dict]:
        """Generate a random quantum circuit"""
        
        # Create quantum and classical registers
        qr = QuantumRegister(num_qubits, 'q')
        cr = ClassicalRegister(num_qubits, 'c')
        circuit = QuantumCircuit(qr, cr)
        
        circuit_info = {
            'qubits': num_qubits,
            'gates': [],
            'entanglement_map': {},
            'depth': 0,
            'width': num_qubits
        }
        
        # Generate random gates
        available_gates = self.config['circuit_parameters']['gate_types']
        
        for gate_num in range(num_gates):
            # Select random gate type
            gate_type = random.choice(available_gates)
            gate_config = self.gate_library.get(gate_type)
            
            if not gate_config:
                continue
            
            # Determine qubit(s) for the gate
            if gate_type in ['cx', 'cz', 'swap']:
                # Two-qubit gates
                control = random.randint(0, num_qubits - 1)
                target = random.choice([q for q in range(num_qubits) if q != control])
                qubits = [control, target]
                
                # Update entanglement map
                key = f"{control}-{target}"
                circuit_info['entanglement_map'][key] = circuit_info['entanglement_map'].get(key, 0) + 1
                
            elif gate_type == 'ccx':
                # Three-qubit gate (Toffoli)
                controls = random.sample(range(num_qubits), 2)
                target = random.choice([q for q in range(num_qubits) if q not in controls])
                qubits = controls + [target]
            else:
                # Single-qubit gates
                qubit = random.randint(0, num_qubits - 1)
                qubits = [qubit]
            
            # Add gate to circuit
            self.add_gate_to_circuit(circuit, gate_type, gate_config, qubits, gate_num)
            
            # Record gate info
            circuit_info['gates'].append({
                'type': gate_type,
                'qubits': qubits,
                'position': gate_num,
                'parameters': self.generate_gate_parameters(gate_config)
            })
        
        # Add measurements at the end
        circuit.measure(range(num_qubits), range(num_qubits))
        
        # Calculate circuit depth
        circuit_info['depth'] = circuit.depth()
        
        return circuit, circuit_info
    
    def add_gate_to_circuit(self, circuit: QuantumCircuit, gate_type: str, 
                           gate_config: Dict, qubits: List[int], gate_num: int):
        """Add a gate to the quantum circuit"""
        
        if gate_type == 'rx':
            angle = random.uniform(0, 2 * math.pi)
            circuit.rx(angle, qubits[0])
        elif gate_type == 'ry':
            angle = random.uniform(0, 2 * math.pi)
            circuit.ry(angle, qubits[0])
        elif gate_type == 'rz':
            angle = random.uniform(0, 2 * math.pi)
            circuit.rz(angle, qubits[0])
        elif gate_type == 'h':
            circuit.h(qubits[0])
        elif gate_type == 'x':
            circuit.x(qubits[0])
        elif gate_type == 'y':
            circuit.y(qubits[0])
        elif gate_type == 'z':
            circuit.z(qubits[0])
        elif gate_type == 's':
            circuit.s(qubits[0])
        elif gate_type == 't':
            circuit.t(qubits[0])
        elif gate_type == 'cx':
            circuit.cx(qubits[0], qubits[1])
        elif gate_type == 'cz':
            circuit.cz(qubits[0], qubits[1])
        elif gate_type == 'swap':
            circuit.swap(qubits[0], qubits[1])
        elif gate_type == 'ccx':
            circuit.ccx(qubits[0], qubits[1], qubits[2])
    
    def generate_gate_parameters(self, gate_config: Dict) -> Dict:
        """Generate parameters for a gate"""
        params = {}
        
        if gate_config['params'] > 0:
            for i in range(gate_config['params']):
                params[f'param_{i}'] = random.uniform(0, 2 * math.pi)
        
        return params
    
    def generate_hidden_message(self, player_id: int, circle: int) -> str:
        """Generate a hidden message for the puzzle"""
        
        # Create message components
        components = [
            f"PLAYER:{player_id}",
            f"CIRCLE:{circle}",
            f"TIME:{int(datetime.now().timestamp())}",
            f"RANDOM:{secrets.token_hex(8)}",
            f"QUANTUM:ENTANGLEMENT"
        ]
        
        # Combine and hash
        message = "|".join(components)
        message_hash = hashlib.sha256(message.encode()).hexdigest()
        
        # Format as flag
        return f"QUANTUM_FLAG_{message_hash[:32].upper()}"
    
    def encode_message_in_circuit(self, circuit: QuantumCircuit, 
                                 message: str) -> QuantumCircuit:
        """Encode a hidden message in the quantum circuit"""
        
        # Convert message to binary
        binary_message = ''.join(format(ord(c), '08b') for c in message)
        
        # Get number of qubits
        num_qubits = circuit.num_qubits
        
        # Create a new circuit with encoded gates
        qr = QuantumRegister(num_qubits, 'q')
        cr = ClassicalRegister(num_qubits, 'c')
        encoded_circuit = QuantumCircuit(qr, cr)
        
        # Copy original gates
        for instruction in circuit.data:
            encoded_circuit.append(instruction[0], instruction[1], instruction[2])
        
        # Add hidden gates based on message bits
        bit_index = 0
        for qubit in range(num_qubits):
            if bit_index < len(binary_message):
                bit = binary_message[bit_index]
                
                # Add a small rotation based on the bit
                if bit == '1':
                    # Add a small, hard-to-notice rotation
                    encoded_circuit.rz(0.01, qubit)
                
                bit_index += 1
        
        return encoded_circuit
    
    def calculate_solution(self, circuit: QuantumCircuit, 
                          hidden_message: str) -> str:
        """Calculate the expected solution for the puzzle"""
        
        # Simulate the circuit
        from qiskit import Aer, execute
        
        backend = Aer.get_backend('qasm_simulator')
        result = execute(circuit, backend, shots=8192).result()
        counts = result.get_counts()
        
        # Find most frequent measurement
        if not counts:
            return "ERROR_NO_MEASUREMENTS"
        
        most_frequent = max(counts, key=counts.get)
        
        # Calculate probabilities
        total_shots = sum(counts.values())
        probabilities = {state: count/total_shots for state, count in counts.items()}
        
        # Create solution based on measurements and hidden message
        solution_data = {
            'most_frequent_state': most_frequent,
            'probability': probabilities[most_frequent],
            'total_states': len(counts),
            'hidden_message_hash': hashlib.sha256(hidden_message.encode()).hexdigest()[:16],
            'entropy': self.calculate_entropy(probabilities)
        }
        
        # Convert to solution string
        solution = self.format_solution(solution_data)
        
        return solution
    
    def calculate_entropy(self, probabilities: Dict[str, float]) -> float:
        """Calculate Shannon entropy of measurement probabilities"""
        entropy = 0.0
        for prob in probabilities.values():
            if prob > 0:
                entropy -= prob * math.log2(prob)
        return entropy
    
    def format_solution(self, solution_data: Dict) -> str:
        """Format solution data as string"""
        # Format: STATE_PROBABILITY_ENTROPY_HASH
        state = solution_data['most_frequent_state']
        prob = f"{solution_data['probability']:.6f}"
        entropy = f"{solution_data['entropy']:.4f}"
        msg_hash = solution_data['hidden_message_hash']
        
        return f"{state}_{prob}_{entropy}_{msg_hash}"
    
    def create_puzzle_data(self, circuit: QuantumCircuit, circuit_info: Dict,
                          player_id: int, circle_number: int, 
                          hidden_message: str) -> Dict:
        """Create complete puzzle data package"""
        
        # Generate QASM representation
        qasm_str = circuit.qasm()
        
        # Generate circuit diagram (ASCII art)
        ascii_diagram = self.generate_ascii_diagram(circuit)
        
        # Generate hints
        hints = self.generate_hints(circuit_info, player_id)
        
        # Create puzzle files
        files = {
            'circuit.qasm': qasm_str,
            'circuit_info.json': json.dumps(circuit_info, indent=2),
            'hints.txt': '\n'.join(hints),
            'ascii_diagram.txt': ascii_diagram,
            'simulator.py': self.generate_simulator_script(),
            'visualization.ipynb': self.generate_visualization_notebook()
        }
        
        # Add hidden message clue (encrypted)
        clue = self.generate_hidden_clue(hidden_message)
        files['clue.enc'] = clue
        
        puzzle_data = {
            'type': 'quantum',
            'title': f'Quantum Entanglement Challenge - Circle {circle_number}',
            'description': self.config['description'],
            'difficulty': self.config['difficulty'],
            'estimated_time': self.config['estimated_time_hours'],
            'files': files,
            'requirements': self.config['required_skills'],
            'learning_objectives': self.config['learning_objectives'],
            'references': self.config['references'],
            'verification': {
                'method': self.config['solution_requirements']['verification_method'],
                'shots_required': self.config['solution_requirements']['measurement_shots'],
                'accuracy_required': self.config['solution_requirements']['required_accuracy']
            }
        }
        
        return puzzle_data
    
    def generate_ascii_diagram(self, circuit: QuantumCircuit) -> str:
        """Generate ASCII art representation of the circuit"""
        try:
            # Try to use Qiskit's text drawer
            from qiskit.visualization import circuit_drawer
            return circuit_drawer(circuit, output='text', fold=-1)
        except:
            # Fallback simple representation
            diagram = []
            diagram.append("=" * 80)
            diagram.append("QUANTUM CIRCUIT DIAGRAM")
            diagram.append("=" * 80)
            
            for i in range(circuit.num_qubits):
                diagram.append(f"q[{i}]: |0⟩───[...{circuit.depth()} gates...]───M───")
            
            diagram.append("")
            diagram.append(f"Total qubits: {circuit.num_qubits}")
            diagram.append(f"Circuit depth: {circuit.depth()}")
            diagram.append(f"Total gates: {len(circuit.data)}")
            
            return '\n'.join(diagram)
    
    def generate_hints(self, circuit_info: Dict, player_id: int) -> List[str]:
        """Generate hints for the puzzle"""
        hints = []
        
        # Basic hints
        hints.append(f"Number of qubits: {circuit_info['qubits']}")
        hints.append(f"Circuit depth: {circuit_info['depth']}")
        
        # Gate distribution hint
        gate_types = [g['type'] for g in circuit_info['gates']]
        gate_counts = {gt: gate_types.count(gt) for gt in set(gate_types)}
        
        hints.append("Gate distribution:")
        for gate_type, count in sorted(gate_counts.items()):
            hints.append(f"  {gate_type}: {count}")
        
        # Entanglement hint
        if circuit_info['entanglement_map']:
            hints.append(f"Entangled pairs: {len(circuit_info['entanglement_map'])}")
        
        # Puzzle-specific hints from config
        hints.extend(self.config['hints'][:3])
        
        # Player-specific hint (encoded)
        player_hint = hashlib.sha256(str(player_id).encode()).hexdigest()[:8]
        hints.append(f"Player code: {player_hint}")
        
        return hints
    
    def generate_simulator_script(self) -> str:
        """Generate Python simulator script"""
        return '''#!/usr/bin/env python3
# Quantum Circuit Simulator for Chimera-VX

import qiskit
from qiskit import QuantumCircuit, Aer, execute
import numpy as np
import matplotlib.pyplot as plt
import json
import sys

def load_circuit(qasm_file):
    """Load quantum circuit from QASM file"""
    with open(qasm_file, 'r') as f:
        qasm_str = f.read()
    return QuantumCircuit.from_qasm_str(qasm_str)

def simulate_circuit(circuit, shots=8192):
    """Simulate quantum circuit"""
    backend = Aer.get_backend('qasm_simulator')
    result = execute(circuit, backend, shots=shots).result()
    return result.get_counts()

def analyze_results(counts):
    """Analyze measurement results"""
    total_shots = sum(counts.values())
    
    print(f"Total shots: {total_shots}")
    print(f"Unique states: {len(counts)}")
    
    # Find most frequent state
    most_frequent = max(counts, key=counts.get)
    probability = counts[most_frequent] / total_shots
    
    print(f"Most frequent state: {most_frequent}")
    print(f"Probability: {probability:.6f}")
    
    # Calculate entropy
    entropy = 0.0
    for count in counts.values():
        prob = count / total_shots
        if prob > 0:
            entropy -= prob * np.log2(prob)
    
    print(f"Shannon entropy: {entropy:.4f}")
    
    return most_frequent, probability, entropy

def visualize_results(counts):
    """Visualize measurement results"""
    states = list(counts.keys())
    frequencies = list(counts.values())
    
    plt.figure(figsize=(12, 6))
    plt.bar(states, frequencies)
    plt.xlabel('Measurement State')
    plt.ylabel('Frequency')
    plt.title('Quantum Measurement Results')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('measurement_results.png')
    print("Visualization saved as 'measurement_results.png'")

def main():
    if len(sys.argv) < 2:
        print("Usage: python simulator.py circuit.qasm [shots]")
        sys.exit(1)
    
    qasm_file = sys.argv[1]
    shots = int(sys.argv[2]) if len(sys.argv) > 2 else 8192
    
    print("=== CHIMERA-VX QUANTUM SIMULATOR ===")
    
    # Load circuit
    circuit = load_circuit(qasm_file)
    print(f"Circuit loaded: {circuit.num_qubits} qubits, depth {circuit.depth()}")
    
    # Simulate
    print(f"Simulating with {shots} shots...")
    counts = simulate_circuit(circuit, shots)
    
    # Analyze
    print("\\n=== ANALYSIS RESULTS ===")
    state, prob, entropy = analyze_results(counts)
    
    # Visualize
    print("\\nGenerating visualization...")
    visualize_results(counts)
    
    print("\\n=== HINTS ===")
    print("1. Look for patterns in the most frequent states")
    print("2. Check entanglement between qubits")
    print("3. The flag may be encoded in measurement probabilities")
    print("4. Use the entropy value as a clue")

if __name__ == "__main__":
    main()
'''
    
    def generate_visualization_notebook(self) -> str:
        """Generate Jupyter notebook for visualization"""
        return '''{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Chimera-VX Quantum Puzzle Analysis\n",
    "## Circle 1: Quantum Entanglement"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Import required libraries\n",
    "from qiskit import QuantumCircuit, Aer, execute\n",
    "from qiskit.visualization import plot_histogram, plot_bloch_multivector, plot_state_city\n",
    "from qiskit.quantum_info import Statevector\n",
    "import matplotlib.pyplot as plt\n",
    "import numpy as np\n",
    "import json"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load the circuit\n",
    "with open('circuit.qasm', 'r') as f:\n",
    "    qasm_str = f.read()\n",
    "    \n",
    "circuit = QuantumCircuit.from_qasm_str(qasm_str)\n",
    "print(f\"Circuit: {circuit.num_qubits} qubits, {circuit.depth()} depth\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Simulate circuit\n",
    "backend = Aer.get_backend('qasm_simulator')\n",
    "result = execute(circuit, backend, shots=8192).result()\n",
    "counts = result.get_counts()\n",
    "\n",
    "print(f\"Total unique states: {len(counts)}\")\n",
    "print(f\"Most frequent state: {max(counts, key=counts.get)}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Visualize measurement results\n",
    "fig = plot_histogram(counts)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Analyze statevector (for small circuits)\n",
    "if circuit.num_qubits <= 8:\n",
    "    backend_sv = Aer.get_backend('statevector_simulator')\n",
    "    result_sv = execute(circuit.remove_final_measurements(inplace=False), backend_sv).result()\n",
    "    statevector = result_sv.get_statevector()\n",
    "    \n",
    "    fig = plot_state_city(statevector)\n",
    "    plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Analysis Tasks\n",
    "1. Identify the most probable measurement outcome\n",
    "2. Calculate the Shannon entropy of the distribution\n",
    "3. Look for entanglement patterns\n",
    "4. Extract the hidden flag from the quantum state"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.8.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}'''
    
    def generate_hidden_clue(self, hidden_message: str) -> str:
        """Generate encrypted clue for the hidden message"""
        # Simple XOR encryption
        key = secrets.token_bytes(16)
        message_bytes = hidden_message.encode()
        
        # XOR encryption
        encrypted = bytes([message_bytes[i] ^ key[i % len(key)] 
                          for i in range(len(message_bytes))])
        
        # Return base64 encoded
        return base64.b64encode(key + encrypted).decode()
    
    def save_puzzle(self, puzzle_data: Dict, output_dir: str = "assets"):
        """Save puzzle to files"""
        import os
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        puzzle_id = puzzle_data['metadata']['puzzle_id']
        puzzle_dir = os.path.join(output_dir, puzzle_id)
        os.makedirs(puzzle_dir, exist_ok=True)
        
        # Save files
        files = puzzle_data['puzzle_data']['files']
        for filename, content in files.items():
            filepath = os.path.join(puzzle_dir, filename)
            with open(filepath, 'w') as f:
                f.write(content)
        
        # Save metadata
        metadata_path = os.path.join(puzzle_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(puzzle_data['metadata'], f, indent=2)
        
        # Save solution (encrypted)
        solution_path = os.path.join(puzzle_dir, 'solution.enc')
        encrypted_solution = self.encrypt_solution(puzzle_data['solution'])
        with open(solution_path, 'w') as f:
            f.write(encrypted_solution)
        
        logger.info(f"Puzzle saved to {puzzle_dir}")
        return puzzle_dir
    
    def encrypt_solution(self, solution: str) -> str:
        """Encrypt solution for storage"""
        # Simple encryption for demo
        key = secrets.token_bytes(32)
        solution_bytes = solution.encode()
        
        encrypted = bytes([solution_bytes[i] ^ key[i % len(key)] 
                          for i in range(len(solution_bytes))])
        
        return base64.b64encode(key + encrypted).decode()


# Test function
def test_generator():
    """Test the quantum puzzle generator"""
    print("Testing Quantum Puzzle Generator...")
    
    generator = QuantumPuzzleGenerator()
    
    # Generate a test puzzle
    puzzle = generator.generate_puzzle(
        player_id=123,
        circle_number=1,
        difficulty_multiplier=1.0
    )
    
    print(f"Puzzle generated successfully!")
    print(f"Puzzle ID: {puzzle['metadata']['puzzle_id']}")
    print(f"Number of qubits: {puzzle['metadata']['qubits']}")
    print(f"Number of gates: {puzzle['metadata']['gates']}")
    print(f"Solution hash: {puzzle['solution_hash'][:32]}...")
    
    # Save puzzle
    puzzle_dir = generator.save_puzzle(puzzle)
    print(f"Puzzle saved to: {puzzle_dir}")
    
    return puzzle


if __name__ == "__main__":
    test_generator()
