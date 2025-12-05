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
        from dataclasses import fields, is_dataclass
        
        program = Program()
        # Use an iterator to consume tokens
        token_iter = iter(sequence)
        
        try:
            while True:
                token = next(token_iter)
                
                if token == 99: # Halt
                    break
                
                # Check for subroutine call
                if token >= 100: # Arbitrary offset for subroutines
                    sub_id = token - 100
                    if self.subroutines.get_subroutine(sub_id):
                        program.add_statement(Statement(SubroutineCall(sub_id)))
                    continue

                # Check for primitive
                if token in self.token_map:
                    primitive_cls = self.token_map[token]
                    
                    # Only Actions can be top-level statements
                    if issubclass(primitive_cls, Action):
                        try:
                            action = self._decode_primitive(primitive_cls, token_iter)
                            program.add_statement(Statement(action))
                        except StopIteration:
                            break
                        except Exception as e:
                            # Skip invalid action construction
                            pass
        except StopIteration:
            pass
            
        return program

    def _decode_primitive(self, cls, token_iter):
        from dataclasses import fields
        
        # Get fields
        cls_fields = fields(cls)
        args = {}
        
        for field in cls_fields:
            # Check field type
            ftype = field.type
            
            # Handle Optional/Union types (simplified: assume first type)
            if hasattr(ftype, '__origin__') and ftype.__origin__ is Union:
                ftype = ftype.__args__[0]
                
            if isinstance(ftype, type) and issubclass(ftype, (Action, Predicate, Filter)):
                # Recursive decode
                # Expect next token to be the class token for this type
                # But wait, the grammar might not enforce explicit type tokens for arguments if they are polymorphic
                # However, for things like 'Filter', it expects a 'Predicate'.
                # We need to peek/consume the next token to see what Predicate it is.
                
                sub_token = next(token_iter)
                if sub_token in self.token_map:
                    sub_cls = self.token_map[sub_token]
                    if issubclass(sub_cls, ftype):
                        args[field.name] = self._decode_primitive(sub_cls, token_iter)
                    else:
                        # Token doesn't match expected type
                        # Fallback: maybe use a default or error
                        # For now, consume args for the WRONG class just to advance?
                        # Or just error out.
                        raise ValueError(f"Expected {ftype}, got {sub_cls}")
                else:
                    raise ValueError(f"Invalid token for {ftype}: {sub_token}")
            else:
                # Simple type (int, str, bool)
                val_token = next(token_iter)
                # Map token to value if needed, or use raw int
                # For now, use raw int. 
                # Ideally we'd map 0-9 to ints, but our tokens ARE ints.
                # If field expects str, we might need a map.
                if ftype is str:
                    args[field.name] = str(val_token)
                elif ftype is bool:
                    args[field.name] = bool(val_token)
                else:
                    args[field.name] = val_token
                    
        return cls(**args)
