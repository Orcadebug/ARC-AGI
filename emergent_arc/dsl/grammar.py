from dataclasses import dataclass, field
from typing import List, Union
from .primitives import Action, Primitive

@dataclass
class Statement:
    action: Action

@dataclass
class Program:
    statements: List[Statement] = field(default_factory=list)

    def add_statement(self, stmt: Statement):
        self.statements.append(stmt)

@dataclass
class SubroutineCall(Action):
    subroutine_id: int

@dataclass
class HaltAction(Action):
    pass
