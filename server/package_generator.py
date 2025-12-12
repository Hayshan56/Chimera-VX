#!/usr/bin/env python3
# chimera-vx/server/package_generator.py
# Puzzle package generator for Chimera-VX

import json
import hashlib
import secrets
import base64
import time
import random
import string
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import logging
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes

logger = logging.getLogger(__name__)

class PackageGenerator:
    """Generate unique puzzle packages for each player"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.puzzle_templates = self.load_puzzle_templates()
        
        # Initialize encryption key
        self.encryption_key = self.generate_encryption_key()
        
        # Puzzle-specific generators
        self.generators = {
            'quantum': QuantumPuzzleGenerator(),
            'dna': DNAPuzzleGenerator(),
            'radio': RadioPuzzleGenerator(),
            'fpga': FPGAPuzzleGenerator(),
            'minecraft': MinecraftPuzzleGenerator(),
            'usb': USBPuzzleGenerator(),
            'temporal': TemporalPuzzleGenerator(),
            'cryptographic': CryptographicPuzzleGenerator(),
            'hardware': HardwarePuzzleGenerator(),
            'forensic': ForensicPuzzleGenerator(),
            'network': NetworkPuzzleGenerator(),
            'meta': MetaPuzzleGenerator()
        }
        
        logger.info("PackageGenerator initialized")
    
    def load_puzzle_templates(self) -> Dict:
        """Load puzzle templates from files"""
        templates = {}
        template_dir = Path(self.config['paths']['puzzle_dir'])
        
        for puzzle_type in self.config['puzzles']['puzzle_order']:
            template_file = template_dir / f"{puzzle_type}_template.json"
            if template_file.exists():
                with open(template_file, 'r') as f:
                    templates[puzzle_type] = json.load(f)
            else:
                templates[puzzle_type] = {
                    'description': f'Default template for {puzzle_type}',
                    'difficulty': 'medium',
                    'estimated_time': 6
                }
        
        return templates
    
    def generate_encryption_key(self) -> bytes:
        """Generate or load encryption key"""
        key_dir = Path(self.config['paths']['key_dir'])
        key_dir.mkdir(exist_ok=True)
        
        key_file = key_dir / "encryption.key"
        
        if key_file.exists():
            with open(key_file, 'rb') as f:
                return f.read()
        else:
            key = secrets.token_bytes(32)
            with open(key_file, 'wb') as f:
                f.write(key)
            logger.info("Generated new encryption key")
            return key
    
    async def generate_initial_package(self, player_id: int) -> Dict:
        """Generate initial package for new player"""
        package = {
            'version': '1.0.0',
            'player_id': player_id,
            'generated_at': int(time.time()),
            'server_info': {
                'name': 'Chimera-VX',
                'total_circles': self.config['puzzles']['total_circles'],
                'estimated_time': self.config['puzzles']['min_solve_time']
            },
            'requirements': {
                'hardware': 'See HARDWARE_REQUIREMENTS.md',
                'software': 'See requirements.txt',
                'time_commitment': '72+ hours recommended'
            },
            'welcome_message': self.generate_welcome_message(),
            'first_challenge': await self.generate_puzzle(player_id, 1, {}),
            'encryption_key': self.generate_player_key(player_id)
        }
        
        # Encrypt sensitive parts
        encrypted_package = self.encrypt_package(package, player_id)
        
        logger.info(f"Generated initial package for player {player_id}")
        return encrypted_package
    
    async def generate_puzzle(self, player_id: int, circle_number: int, 
                             player_data: Dict) -> Dict:
        """Generate a specific puzzle for a player"""
        puzzle_type = self.config['puzzles']['puzzle_order'][circle_number - 1]
        
        logger.info(f"Generating {puzzle_type} puzzle for player {player_id}")
        
        # Get generator for this puzzle type
        generator = self.generators.get(puzzle_type)
        if not generator:
            raise ValueError(f"No generator for puzzle type: {puzzle_type}")
        
        # Generate unique seed for this player+puzzle
        seed = f"{player_id}_{circle_number}_{player_data.get('hardware_fingerprint', '')}_{int(time.time())}"
        random.seed(hashlib.sha256(seed.encode()).hexdigest())
        
        # Generate puzzle
        puzzle_data = await generator.generate(
            player_id=player_id,
            circle_number=circle_number,
            player_data=player_data,
            config=self.config
        )
        
        # Calculate solution hash
        solution_hash = hashlib.sha256(
            puzzle_data['solution'].encode()
        ).hexdigest()
        
        # Add metadata
        puzzle_data['metadata'] = {
            'player_id': player_id,
            'circle': circle_number,
            'type': puzzle_type,
            'generated_at': int(time.time()),
            'unique_id': secrets.token_hex(8),
            'expected_solve_time': self.puzzle_templates[puzzle_type].get('estimated_time', 6)
        }
        
        return {
            'puzzle': puzzle_data,
            'solution_hash': solution_hash,
            'type': puzzle_type,
            'circle': circle_number
        }
    
    async def generate_final_flag(self, player_id: int) -> str:
        """Generate final flag for completing all circles"""
        # Get all solution hashes for this player
        solution_hashes = []
        for circle in range(1, 13):
            seed = f"{player_id}_{circle}_final"
            random.seed(hashlib.sha256(seed.encode()).hexdigest())
            fake_solution = secrets.token_hex(32)
            solution_hash = hashlib.sha256(fake_solution.encode()).hexdigest()
            solution_hashes.append(solution_hash)
        
        # Build Merkle tree
        merkle_root = self.build_merkle_tree(solution_hashes)
        
        # Generate final flag
        timestamp = int(time.time())
        flag_data = f"CHIMERA-VX-PLAYER-{player_id}-ROOT-{merkle_root}-TIME-{timestamp}"
        flag_hash = hashlib.sha256(flag_data.encode()).hexdigest()
        
        final_flag = f"CHIMERA{{{flag_hash}}}"
        
        logger.info(f"Generated final flag for player {player_id}")
        return final_flag
    
    def generate_welcome_message(self) -> str:
        """Generate epic welcome message"""
        messages = [
            "Welcome to Chimera-VX. Turn back now.",
            "The abyss stares back. Are you ready?",
            "This is not a game. This is a trial.",
            "72 hours of hell await. Proceed with caution.",
            "When AI fails, humans endure. Begin your journey.",
            "The only way out is through. All 12 circles.",
            "Your sanity will be tested. Your skills will be pushed.",
            "Welcome to the ultimate cybersecurity challenge.",
            "This will break you. That's the point.",
            "The flag is not the reward. The suffering is."
        ]
        
        return random.choice(messages)
    
    def generate_player_key(self, player_id: int) -> str:
        """Generate player-specific encryption key"""
        # Derive key from player ID and server key
        info = f"player_{player_id}_key".encode()
        kdf = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=self.encryption_key[:16],
            info=info
        )
        key = kdf.derive(self.encryption_key)
        return base64.b64encode(key).decode()
    
    def encrypt_package(self, package: Dict, player_id: int) -> Dict:
        """Encrypt sensitive parts of package"""
        # Convert to JSON
        package_json = json.dumps(package).encode()
        
        # Use player-specific key
        player_key = base64.b64decode(package['encryption_key'])
        
        # Encrypt with ChaCha20-Poly1305
        chacha = ChaCha20Poly1305(player_key)
        nonce = secrets.token_bytes(12)
        encrypted = chacha.encrypt(nonce, package_json, None)
        
        return {
            'encrypted': base64.b64encode(nonce + encrypted).decode(),
            'player_id': player_id,
            'encryption_type': 'ChaCha20-Poly1305',
            'key_derivation': 'HKDF-SHA256'
        }
    
    def build_merkle_tree(self, hashes: List[str]) -> str:
        """Build Merkle tree from solution hashes"""
        if len(hashes) == 1:
            return hashes[0]
        
        next_level = []
        for i in range(0, len(hashes), 2):
            if i + 1 < len(hashes):
                combined = hashes[i] + hashes[i + 1]
            else:
                combined = hashes[i] + hashes[i]  # Duplicate for odd number
            next_hash = hashlib.sha256(combined.encode()).hexdigest()
            next_level.append(next_hash)
        
        return self.build_merkle_tree(next_level)
    
    def generate_unique_id(self, player_id: int, puzzle_type: str) -> str:
        """Generate unique ID for puzzle instance"""
        timestamp = int(time.time())
        random_part = secrets.token_hex(4)
        return f"{player_id}_{puzzle_type}_{timestamp}_{random_part}"


# ==================== PUZZLE GENERATOR CLASSES ====================

class BasePuzzleGenerator:
    """Base class for all puzzle generators"""
    
    async def generate(self, player_id: int, circle_number: int, 
                      player_data: Dict, config: Dict) -> Dict:
        """Generate puzzle - to be implemented by subclasses"""
        raise NotImplementedError


class QuantumPuzzleGenerator(BasePuzzleGenerator):
    """Generate quantum computing puzzles"""
    
    async def generate(self, player_id: int, circle_number: int, 
                      player_data: Dict, config: Dict) -> Dict:
        """Generate quantum puzzle"""
        
        # Create unique quantum circuit
        num_qubits = 4 + circle_number
        gates = self.generate_gates(num_qubits, player_id)
        
        # Generate QASM code
        qasm = self.generate_qasm(num_qubits, gates, player_id)
        
        # Create expected solution
        solution = self.calculate_solution(gates, player_id)
        
        # Additional files
        extra_files = {
            'quantum_simulator.py': self.generate_simulator_script(qasm),
            'visualization.ipynb': self.generate_visualization(qasm),
            'hints.txt': self.generate_hints(circle_number)
        }
        
        return {
            'type': 'quantum',
            'title': f'Quantum Entanglement - Circle {circle_number}',
            'description': 'Simulate the quantum circuit and extract the hidden message.',
            'files': {
                'circuit.qasm': qasm,
                **extra_files
            },
            'requirements': [
                'Python 3.8+',
                'Qiskit library',
                'Quantum simulator (local or IBM Quantum)'
            ],
            'solution': solution,
            'verification_hint': 'The solution is in the measurement probabilities.'
        }
    
    def generate_gates(self, num_qubits: int, player_id: int) -> List[Dict]:
        """Generate random quantum gates"""
        gates = []
        gate_types = ['h', 'x', 'y', 'z', 'cx', 'rx', 'ry', 'rz']
        
        # Seed randomness with player ID
        random.seed(player_id)
        
        for _ in range(10 + num_qubits * 2):
            gate_type = random.choice(gate_types)
            qubit = random.randint(0, num_qubits - 1)
            
            if gate_type in ['cx']:
                control = qubit
                target = (qubit + 1) % num_qubits
                gates.append({'type': gate_type, 'control': control, 'target': target})
            elif gate_type in ['rx', 'ry', 'rz']:
                angle = random.uniform(0, 2 * 3.14159)
                gates.append({'type': gate_type, 'qubit': qubit, 'angle': angle})
            else:
                gates.append({'type': gate_type, 'qubit': qubit})
        
        return gates
    
    def generate_qasm(self, num_qubits: int, gates: List[Dict], player_id: int) -> str:
        """Generate QASM code"""
        lines = [
            'OPENQASM 2.0;',
            'include "qelib1.inc";',
            f'qreg q[{num_qubits}];',
            f'creg c[{num_qubits}];',
            ''
        ]
        
        for gate in gates:
            if gate['type'] == 'cx':
                lines.append(f'cx q[{gate["control"]}], q[{gate["target"]}];')
            elif gate['type'] in ['rx', 'ry', 'rz']:
                lines.append(f'{gate["type"]}({gate["angle"]}) q[{gate["qubit"]}];')
            else:
                lines.append(f'{gate["type"]} q[{gate["qubit"]}];')
        
        lines.append('')
        lines.append('measure q -> c;')
        
        return '\n'.join(lines)
    
    def calculate_solution(self, gates: List[Dict], player_id: int) -> str:
        """Calculate expected solution"""
        # Simplified solution calculation
        # In real implementation, this would simulate the circuit
        gate_string = ''.join(g['type'][0] for g in gates[:8])
        solution_hash = hashlib.sha256(gate_string.encode()).hexdigest()
        return solution_hash[:16]
    
    def generate_simulator_script(self, qasm: str) -> str:
        """Generate Python simulator script"""
        return f'''#!/usr/bin/env python3
# Quantum Circuit Simulator

import qiskit
from qiskit import QuantumCircuit, Aer, execute
import numpy as np

# Load the circuit
circuit = QuantumCircuit.from_qasm_str(\"\"\"{qasm}\"\"\")

# Simulate
backend = Aer.get_backend('qasm_simulator')
result = execute(circuit, backend, shots=8192).result()
counts = result.get_counts()

print("Measurement counts:")
for state, count in counts.items():
    print(f"  {{state}}: {{count}}")

# Hint: The flag is hidden in the most frequent measurement outcomes
'''

    def generate_visualization(self, qasm: str) -> str:
        """Generate Jupyter notebook for visualization"""
        return '''{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Quantum Circuit Visualization"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from qiskit import QuantumCircuit, Aer, execute\\n",
    "from qiskit.visualization import plot_histogram, plot_bloch_multivector\\n",
    "import matplotlib.pyplot as plt"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}'''
    
    def generate_hints(self, circle_number: int) -> str:
        """Generate hints for the puzzle"""
        hints = [
            "Use at least 8192 shots for accurate probabilities.",
            "The most frequent measurement contains the flag.",
            "Convert binary measurements to ASCII.",
            "Some gates may be decoys.",
            "Look for patterns in the measurement distribution."
        ]
        return '\n'.join(hints[:circle_number])


class DNAPuzzleGenerator(BasePuzzleGenerator):
    """Generate DNA sequencing puzzles"""
    
    async def generate(self, player_id: int, circle_number: int, 
                      player_data: Dict, config: Dict) -> Dict:
        """Generate DNA puzzle"""
        
        # Generate DNA sequence with hidden message
        sequence = self.generate_dna_sequence(player_id, circle_number)
        
        # Create FASTQ file
        fastq = self.generate_fastq(sequence, player_id)
        
        # Generate solution
        solution = self.extract_hidden_message(sequence)
        
        # Additional files
        extra_files = {
            'reference.fasta': self.generate_reference_sequence(),
            'analysis.py': self.generate_analysis_script(),
            'protocol.md': self.generate_protocol(circle_number)
        }
        
        return {
            'type': 'dna',
            'title': f'DNA Cipher - Circle {circle_number}',
            'description': 'Analyze the DNA sequencing data to find the hidden message.',
            'files': {
                'sample.fastq': fastq,
                **extra_files
            },
            'requirements': [
                'Python 3.8+',
                'Biopython library',
                'Basic bioinformatics knowledge'
            ],
            'solution': solution,
            'verification_hint': 'The message is encoded in specific nucleotide patterns.'
        }
    
    def generate_dna_sequence(self, player_id: int, circle_number: int) -> str:
        """Generate DNA sequence with hidden message"""
        bases = ['A', 'C', 'G', 'T']
        
        # Create hidden message
        message = f"CTF-SECRET-{player_id}-CIRCLE-{circle_number}"
        binary_message = ''.join(format(ord(c), '08b') for c in message)
        
        # Encode binary in DNA (00=A, 01=C, 10=G, 11=T)
        dna_map = {'00': 'A', '01': 'C', '10': 'G', '11': 'T'}
        encoded = ''
        for i in range(0, len(binary_message), 2):
            chunk = binary_message[i:i+2]
            encoded += dna_map.get(chunk, 'A')
        
        # Add noise and padding
        sequence = ''
        for base in encoded:
            sequence += base
            # Add random bases as noise
            for _ in range(random.randint(1, 3)):
                sequence += random.choice(bases)
        
        return sequence
    
    def generate_fastq(self, sequence: str, player_id: int) -> str:
        """Generate FASTQ file"""
        lines = []
        read_length = 100
        
        for i in range(0, len(sequence), read_length):
            read = sequence[i:i+read_length]
            quality = ''.join(random.choices(['F', ':', ',', '#'], k=len(read)))
            
            lines.append(f'@READ_{player_id}_{i}')
            lines.append(read)
            lines.append('+')
            lines.append(quality)
        
        return '\n'.join(lines)
    
    def extract_hidden_message(self, sequence: str) -> str:
        """Extract hidden message from DNA sequence"""
        # Remove noise (every 4th base contains the message)
        encoded = sequence[::4]
        
        # Decode DNA to binary
        dna_to_bin = {'A': '00', 'C': '01', 'G': '10', 'T': '11'}
        binary = ''.join(dna_to_bin.get(base, '00') for base in encoded)
        
        # Convert binary to ASCII
        message = ''
        for i in range(0, len(binary), 8):
            byte = binary[i:i+8]
            if len(byte) == 8:
                message += chr(int(byte, 2))
        
        return message.split('\x00')[0]  # Remove null padding
    
    def generate_reference_sequence(self) -> str:
        """Generate reference sequence"""
        return """>reference_sequence
ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG
GCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTA
"""
    
    def generate_analysis_script(self) -> str:
        """Generate analysis script"""
        return '''#!/usr/bin/env python3
# DNA Sequence Analyzer

from Bio import SeqIO
import collections

def analyze_fastq(fastq_file):
    """Analyze FASTQ file"""
    sequences = []
    qualities = []
    
    for record in SeqIO.parse(fastq_file, "fastq"):
        sequences.append(str(record.seq))
        qualities.append(record.letter_annotations["phred_quality"])
    
    print(f"Total reads: {len(sequences)}")
    print(f"Average length: {sum(len(s) for s in sequences) / len(sequences)}")
    
    # Look for patterns
    base_counts = collections.Counter()
    for seq in sequences:
        base_counts.update(seq)
    
    print("\\nBase frequencies:")
    for base in "ACGT":
        print(f"  {base}: {base_counts.get(base, 0)}")
    
    return sequences

if __name__ == "__main__":
    sequences = analyze_fastq("sample.fastq")
    print("\\nHint: Look for patterns in the sequence...")
'''
    
    def generate_protocol(self, circle_number: int) -> str:
        """Generate analysis protocol"""
        return f'''# DNA Analysis Protocol - Circle {circle_number}

  ## Steps:
1. Quality control of FASTQ data
2. Sequence alignment (if reference provided)
3. Pattern analysis
4. Error correction
5. Message extraction

## Tools:
- Biopython for sequence manipulation
- Custom Python scripts for analysis
- Basic statistical analysis

## Hints:
- The message is encoded in specific positions
- Noise has been added to obscure the message
- Look for non-random patterns in base distribution
'''


# Other puzzle generators would follow similar patterns
# For brevity, I'll show the structure but not complete implementations

class RadioPuzzleGenerator(BasePuzzleGenerator):
    """Generate radio signal puzzles"""
    async def generate(self, player_id: int, circle_number: int, 
                      player_data: Dict, config: Dict) -> Dict:
        return {
            'type': 'radio',
            'title': f'Radio Signal Analysis - Circle {circle_number}',
            'description': 'Decode the hidden transmission from IQ data.',
            'files': {},
            'solution': f'RADIO-SECRET-{player_id}',
            'verification_hint': 'Multiple modulation schemes are used.'
        }


class FPGAPuzzleGenerator(BasePuzzleGenerator):
    """Generate FPGA puzzles"""
    async def generate(self, player_id: int, circle_number: int, 
                      player_data: Dict, config: Dict) -> Dict:
        return {
            'type': 'fpga',
            'title': f'FPGA Bitstream Analysis - Circle {circle_number}',
            'description': 'Reverse engineer the Verilog code.',
            'files': {},
            'solution': f'FPGA-SECRET-{player_id}',
            'verification_hint': 'Look for hardware trojans.'
        }


class MinecraftPuzzleGenerator(BasePuzzleGenerator):
    """Generate Minecraft puzzles"""
    async def generate(self, player_id: int, circle_number: int, 
                      player_data: Dict, config: Dict) -> Dict:
        return {
            'type': 'minecraft',
            'title': f'Minecraft Computer - Circle {circle_number}',
            'description': 'Extract the program from the redstone computer.',
            'files': {},
            'solution': f'MINECRAFT-SECRET-{player_id}',
            'verification_hint': 'The computer implements a custom CPU.'
        }


class USBPuzzleGenerator(BasePuzzleGenerator):
    """Generate USB protocol puzzles"""
    async def generate(self, player_id: int, circle_number: int, 
                      player_data: Dict, config: Dict) -> Dict:
        return {
            'type': 'usb',
            'title': f'USB Protocol Analysis - Circle {circle_number}',
            'description': 'Reverse engineer the USB device protocol.',
            'files': {},
            'solution': f'USB-SECRET-{player_id}',
            'verification_hint': 'Multiple endpoints with different functions.'
        }


class TemporalPuzzleGenerator(BasePuzzleGenerator):
    """Generate temporal puzzles"""
    async def generate(self, player_id: int, circle_number: int, 
                      player_data: Dict, config: Dict) -> Dict:
        return {
            'type': 'temporal',
            'title': f'Temporal Paradox - Circle {circle_number}',
            'description': 'Solve time-dependent puzzles.',
            'files': {},
            'solution': f'TEMPORAL-SECRET-{player_id}',
            'verification_hint': 'Solutions change with time.'
        }


class CryptographicPuzzleGenerator(BasePuzzleGenerator):
    """Generate cryptographic puzzles"""
    async def generate(self, player_id: int, circle_number: int, 
                      player_data: Dict, config: Dict) -> Dict:
        return {
            'type': 'cryptographic',
            'title': f'Cryptographic Maze - Circle {circle_number}',
            'description': 'Break the custom cryptosystem.',
            'files': {},
            'solution': f'CRYPTO-SECRET-{player_id}',
            'verification_hint': 'Multiple layers of encryption.'
        }


class HardwarePuzzleGenerator(BasePuzzleGenerator):
    """Generate hardware puzzles"""
    async def generate(self, player_id: int, circle_number: int, 
                      player_data: Dict, config: Dict) -> Dict:
        return {
            'type': 'hardware',
            'title': f'Hardware Glitching - Circle {circle_number}',
            'description': 'Extract secrets via side-channel analysis.',
            'files': {},
            'solution': f'HARDWARE-SECRET-{player_id}',
            'verification_hint': 'Power traces contain the key.'
        }


class ForensicPuzzleGenerator(BasePuzzleGenerator):
    """Generate forensic puzzles"""
    async def generate(self, player_id: int, circle_number: int, 
                      player_data: Dict, config: Dict) -> Dict:
        return {
            'type': 'forensic',
            'title': f'Forensic Abyss - Circle {circle_number}',
            'description': 'Reconstruct evidence from corrupted data.',
            'files': {},
            'solution': f'FORENSIC-SECRET-{player_id}',
            'verification_hint': 'Multiple filesystems, partial data.'
        }


class NetworkPuzzleGenerator(BasePuzzleGenerator):
    """Generate network puzzles"""
    async def generate(self, player_id: int, circle_number: int, 
                      player_data: Dict, config: Dict) -> Dict:
        return {
            'type': 'network',
            'title': f'Network Nexus - Circle {circle_number}',
            'description': 'Hack through custom network stack.',
            'files': {},
            'solution': f'NETWORK-SECRET-{player_id}',
            'verification_hint': 'Custom protocol with encryption.'
        }


class MetaPuzzleGenerator(BasePuzzleGenerator):
    """Generate meta puzzles"""
    async def generate(self, player_id: int, circle_number: int, 
                      player_data: Dict, config: Dict) -> Dict:
        return {
            'type': 'meta',
            'title': 'Meta Synthesis - Final Circle',
            'description': 'Combine all solutions into final flag.',
            'files': {},
            'solution': f'META-SECRET-{player_id}',
            'verification_hint': 'Merkle tree of all solutions.'
        }


# Test the generator
if __name__ == "__main__":
    import asyncio
    
    config = {
        'puzzles': {
            'puzzle_order': ['quantum', 'dna', 'radio', 'fpga', 'minecraft',
                           'usb', 'temporal', 'cryptographic', 'hardware',
                           'forensic', 'network', 'meta'],
            'total_circles': 12
        },
        'paths': {
            'puzzle_dir': 'puzzles',
            'key_dir': 'keys'
        }
    }
    
    generator = PackageGenerator(config)
    
    async def test():
        puzzle = await generator.generate_puzzle(1, 1, {'hardware_fingerprint': 'test'})
        print(f"Generated puzzle type: {puzzle['type']}")
        print(f"Solution hash: {puzzle['solution_hash']}")
    
    asyncio.run(test())
```

