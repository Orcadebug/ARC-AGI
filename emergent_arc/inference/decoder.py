from typing import List, Any, Dict, Tuple
import numpy as np
from ..dsl.grammar import Program, Statement, Action, SubroutineCall, HaltAction
from ..dsl.primitives import *
from ..memory.subroutines import SubroutineLibrary

class ProgramDecoder:
    def __init__(self, subroutine_library: SubroutineLibrary):
        self.subroutines = subroutine_library
        # Map token indices to primitive classes or control tokens
        # This is a simplified mapping. In a real system, this would be dynamic or learned.
        self.token_map = self._build_token_map()

    def _build_token_map(self) -> Dict[int, Any]:
        # Define a fixed mapping for demonstration
        # 0-9: Digits/Arguments
        # 10-19: Tier 1 Actions
        # 20-29: Tier 2 Control/Predicates
        # 30-39: Tier 3 Generative
        # 40+: Subroutines (dynamic)
        mapping = {
            10: Rotate, 11: Flip, 12: Translate, 13: Gravity, 14: Scale, 15: Delete, 16: Clone,
            20: Filter, 21: Foreach, 22: If, 23: SortObjects, 24: GroupBy,
            30: FloodFill, 31: DrawLine, 32: DrawRect, 33: ExtrudePattern, 34: MirrorAcross, 35: CompleteSymmetry, 36: ConnectObjects, 37: Crop, 38: Paste, 39: Recolor,
            99: HaltAction
        }
        return mapping

    def decode_sequence(self, sequence: List[int]) -> Program:
        """
        Decodes a flat sequence of integers into a structured Program.
        """
        program = Program()
        idx = 0
        while idx < len(sequence):
            token = sequence[idx]
            
            if token == 99: # Halt
                break
            
            # Check for subroutine call
            if token >= 100: # Arbitrary offset for subroutines
                sub_id = token - 100
                if self.subroutines.get_subroutine(sub_id):
                    program.add_statement(Statement(SubroutineCall(sub_id)))
                idx += 1
                continue

            # Check for primitive
            if token in self.token_map:
                primitive_cls = self.token_map[token]
                
                # Handle control flow structures recursively
                if primitive_cls == Foreach:
                    # Foreach(filter, action)
                    # Expect next tokens to define filter and action
                    idx += 1
                    if idx >= len(sequence): break
                    
                    # Decode Filter
                    filter_token = sequence[idx]
                    # ... (Simplified: assume next token is filter predicate)
                    # In full implementation, we'd need a more robust parser
                    # For now, let's just skip complex nesting in this mock
                    pass
                elif primitive_cls == If:
                    # If(cond, then, else)
                    pass
                else:
                    # Simple Action
                    # Parse arguments based on dataclass fields
                    # This requires introspection or a predefined schema
                    # For this demo, we'll just instantiate with dummy args or next tokens
                    try:
                        # Mock argument parsing: take next N tokens as args
                        # num_args = len(fields(primitive_cls))
                        # args = sequence[idx+1 : idx+1+num_args]
                        # action = primitive_cls(*args)
                        # program.add_statement(Statement(action))
                        # idx += 1 + num_args
                        
                        # Simplified: Just add the class type for now to show structure
                        # In real system, we need robust argument decoding
                        pass
                    except:
                        pass
            
            idx += 1
            
        return program

    def decode_weights_to_program(self, weights: Any) -> Program:
        """
        Decodes continuous weights (from SNN/GA) into a Program.
        Usually involves argmax to get tokens, then decode_sequence.
        """
        # Mock implementation
        # Assume weights is already a sequence of tokens for this step
        if isinstance(weights, list) or isinstance(weights, np.ndarray):
            return self.decode_sequence(list(weights))
        return Program()
