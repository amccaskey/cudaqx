import sys
import argparse
import importlib.util
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum

from qkec.mlir.ir import Context, Module, Operation, Value, Block
from qkec.mlir.dialects import builtin, func, quake, cc

@dataclass
class QuantumOperation:
    """Parsed quantum operation with all relevant information"""
    name: str                    # e.g., "quake.h", "quake.x"
    gate_name: str              # e.g., "h", "x", "cnot"
    operands: List[Value]         # All operand SSA values
    results: List[Value]          # All result SSA values
    controls: List[Value]         # Control qubit SSA values
    targets: List[Value]          # Target qubit SSA values
    parameters: List[Value]       # Parameter SSA namvalueses (for parameterized gates)
    is_adjoint: bool           # Whether this is an adjoint operation
    is_controlled: bool        # Whether this gate has controls
    num_targets: int           # Number of target qubits
    num_controls: int          # Number of control qubits
    attributes: Dict[str, Any] # Additional attributes


class QuakeVisitorException(Exception):
    """Base exception for visitor errors"""
    pass


class UnimplementedOperationException(QuakeVisitorException):
    """Raised when a visit method is not implemented for an operation"""
    def __init__(self, operation: str):
        super().__init__(f"Visit method not implemented for operation: {operation}")
        self.operation = operation


class BaseQuakeVisitor(ABC):
    """
    Improved base visitor for CUDA-Q Quake dialect operations.
    Handles wire mapping automatically and provides a generic quantum operation interface.
    """
    
    def __init__(self):
        self.wire_map: Dict[Value, int] = {}  # Maps SSA values to qubit indices
        self.qubit_counter = 0
        self.visited_operations = []
        
    def visit_module(self, module_op: Operation) -> Any:
        """Main entry point for visiting a ModuleOp"""
        # Initialize translation
        self.begin_translation()
        
        # Find and visit function operations
        for op in module_op.regions[0].blocks[0].operations:
            if isinstance(op, func.FuncOp):
                self._visit_function(op)
                
        # Finalize translation  
        return self.end_translation()
    
    def _visit_function(self, func_op: Operation):
        """Visit a function operation and its body"""
        # Check if this is a CUDA-Q kernel
        attrs = {str(attr.name): attr.attr for attr in func_op.attributes}
        if "cudaq-kernel" not in attrs:
            return  # Skip non-kernel functions
        
        # Visit function body
        if func_op.regions:
            for region in func_op.regions:
                for block in region:
                    for op in block:
                        if op.OPERATION_NAME.startswith("quake."):
                            self._dispatch_operation(op)
    
    def _dispatch_operation(self, op: Operation):
        """Dispatch operation to appropriate visit method"""
        if op.OPERATION_NAME.startswith("quake.") and self._is_quantum_gate(op.OPERATION_NAME):
            # Handle quantum operations generically
            quantum_op = self._parse_quantum_operation(op)
            self._update_wire_mappings(quantum_op)
            self.visit_quantum_operation(quantum_op)
        else:
            # Handle non-quantum operations with specific methods
            method_name = f"visit_{op.OPERATION_NAME.replace('.', '_')}"
            if hasattr(self, method_name):
                method = getattr(self, method_name)
                method(op)
            else:
                raise UnimplementedOperationException(op.OPERATION_NAME)
    
    def _is_quantum_gate(self, op_name: str) -> bool:
        """Check if an operation is a quantum gate/operation"""
        quantum_gates = {
            'quake.h', 'quake.x', 'quake.y', 'quake.z', 'quake.s', 'quake.t',
            'quake.rx', 'quake.ry', 'quake.rz', 'quake.r1', 'quake.u2', 'quake.u3',
            'quake.swap', 'quake.exp_pauli', 'quake.phased_rx', 'quake.custom_op',
            'quake.mz', 'quake.mx', 'quake.my'
        }
        return op_name in quantum_gates
    
    def _parse_quantum_operation(self, op: Operation) -> QuantumOperation:
        """Parse an MLIR quantum operation into a QuantumOperation object"""
        op_name = op.OPERATION_NAME
        gate_name = op_name.split('.')[-1]  # Extract gate name (e.g., 'h' from 'quake.h')
        
        # Extract basic information
        operands = op.operands #self._extract_operands(op)
        results = op.results #self._extract_results(op)
        
        # Parse attributes
        attributes = {}
        is_adjoint = False
        for attr in op.attributes:
            attr_name = str(attr.name)
            if attr_name == "is_adj":
                is_adjoint = bool(attr.attr)
            attributes[attr_name] = attr.attr
        
        # Parse controls, targets, and parameters based on operation structure
        controls, targets, parameters = self._parse_operand_structure(op, operands)
        
        # Handle controlled operations specially
        if controls and gate_name == 'x' and len(targets) == 1:
            gate_name = 'cnot'  # Special case for controlled X
        elif controls:
            gate_name = f'c{gate_name}'  # Prefix with 'c' for controlled
        
        return QuantumOperation(
            name=op_name,
            gate_name=gate_name,
            operands=operands,
            results=results,
            controls=controls,
            targets=targets,
            parameters=parameters,
            is_adjoint=is_adjoint,
            is_controlled=len(controls) > 0,
            num_targets=len(targets),
            num_controls=len(controls),
            attributes=attributes
        )
    
    def _parse_operand_structure(self, op: Operation, operands: List[Value]) -> tuple[List[Value], List[Value], List[Value]]:
        """Parse operands into controls, targets, and parameters"""
        controls = []
        targets = []
        parameters = []
        
        # This is a simplified parsing - real implementation would need to analyze
        # the operation's structure more carefully based on the tablegen definitions
        
        # For measurements, all operands are targets
        if op.OPERATION_NAME in ['quake.mz', 'quake.mx', 'quake.my']:
            targets = operands
        # For most single-qubit gates, the operand is a target
        elif op.OPERATION_NAME in ['quake.h', 'quake.x', 'quake.y', 'quake.z', 'quake.s', 'quake.t']:
            if len(operands) == 1:
                targets = operands
            elif len(operands) == 2:
                # Likely controlled version
                controls = operands[:1]
                targets = operands[1:]
        # For parameterized gates, need to distinguish parameters from qubits
        elif op.OPERATION_NAME in ['quake.rx', 'quake.ry', 'quake.rz', 'quake.r1']:
            # First operand is typically parameter, last is target
            if len(operands) >= 2:
                parameters = operands[:-1]
                targets = operands[-1:]
            else:
                targets = operands
        # For two-qubit gates like swap
        elif op.OPERATION_NAME == 'quake.swap':
            targets = operands
        else:
            # Default: assume all operands are targets
            targets = operands
            
        return controls, targets, parameters
    
    def _update_wire_mappings(self, quantum_op: QuantumOperation):
        """Update wire mappings automatically for quantum operations"""
        # Map all target and control qubits
        all_qubits = [c for c in quantum_op.controls]
        all_qubits += [t for t in quantum_op.targets]
        
        for qubit_wire in all_qubits:
            if qubit_wire not in self.wire_map:
                self.wire_map[qubit_wire] = self.qubit_counter
                self.qubit_counter += 1
        
        # Update wire mappings for results
        for i, result in enumerate(quantum_op.results):
            if i < len(all_qubits):
                # Result corresponds to input qubit
                self.wire_map[result] = self.wire_map[all_qubits[i]]
    
    def get_qubit_indices(self, wire_names: List[str]) -> List[int]:
        """Helper method to get qubit indices for a list of wire names"""
        return [self.wire_map.get(wire, -1) for wire in wire_names]
    
    def _get_ssa_name(self, value: Value) -> str:
        """Get SSA name from a Value object"""
        value_str = str(value)
        if '%' in value_str:
            parts = value_str.split()
            for part in parts:
                if part.startswith('%'):
                    return part
        return f"%{id(value)}"  # Fallback to object ID
    
    def _extract_operands(self, op: Operation) -> List[str]:
        """Extract operand SSA names from an operation"""
        return [self._get_ssa_name(operand) for operand in op.operands]
    
    def _extract_results(self, op: Operation) -> List[str]:
        """Extract result SSA names from an operation"""
        return [self._get_ssa_name(result) for result in op.results]
    
    # Abstract methods that subclasses must implement
    @abstractmethod
    def begin_translation(self):
        """Initialize the translation (e.g., output headers)"""
        pass
    
    @abstractmethod
    def end_translation(self) -> Any:
        """Finalize the translation and return result"""
        pass
    
    @abstractmethod
    def visit_quantum_operation(self, op: QuantumOperation):
        """
        Generic quantum operation visitor - subclasses must implement this.
        The base class handles wire mapping automatically.
        """
        pass
    
    # Specific operation handlers (can still be overridden for special cases)
    def visit_quake_null_wire(self, op: Operation):
        """Handle wire allocation - subclasses can override if needed"""
        results = op.results #self._extract_results(op)
        for result in results:
            if result not in self.wire_map:
                self.wire_map[result] = self.qubit_counter
                self.qubit_counter += 1
        
        # Call hook for subclasses
        self.handle_wire_allocation(results)
    
    def handle_wire_allocation(self, wire_names: List[str]):
        """Hook for subclasses to handle wire allocation"""
        pass  # Default: do nothing
