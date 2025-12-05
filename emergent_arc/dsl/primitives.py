from dataclasses import dataclass
from typing import List, Any, Optional, Union, Callable

# --- Base Classes ---

@dataclass
class Primitive:
    pass

@dataclass
class Action(Primitive):
    pass

@dataclass
class Predicate(Primitive):
    pass

# --- Tier 1: Rigid Body Primitives ---

@dataclass
class Rotate(Action):
    object_id: int # 0-15
    degrees: int # 90, 180, 270

@dataclass
class Flip(Action):
    object_id: int
    axis: int # 0: horizontal, 1: vertical

@dataclass
class Translate(Action):
    object_id: int
    dx: int
    dy: int

@dataclass
class Gravity(Action):
    direction: int # 0: up, 1: down, 2: left, 3: right

@dataclass
class Scale(Action):
    object_id: int
    factor: int

@dataclass
class Delete(Action):
    object_id: int

@dataclass
class Clone(Action):
    object_id: int
    offset_x: int
    offset_y: int

# --- Tier 2: Relational Predicates & Control ---

@dataclass
class Filter(Primitive):
    predicate: Predicate

@dataclass
class Foreach(Action):
    object_set: Filter
    action: Action

@dataclass
class If(Action):
    condition: Predicate
    then_action: Action
    else_action: Action

@dataclass
class RelativePos(Primitive):
    obj_a: int
    obj_b: int

@dataclass
class SortObjects(Action):
    attribute: str # 'area', 'color', 'x', 'y'
    order: str # 'asc', 'desc'

@dataclass
class GroupBy(Action):
    attribute: str

@dataclass
class Nearest(Primitive):
    object_id: int

@dataclass
class Count(Primitive):
    predicate: Predicate

@dataclass
class Inside(Predicate):
    obj_a: int
    obj_b: int

@dataclass
class Outside(Predicate):
    obj_a: int
    obj_b: int

@dataclass
class AdjacentTo(Predicate):
    obj_a: int
    obj_b: int

@dataclass
class AlignedWith(Predicate):
    obj_a: int
    obj_b: int
    axis: int # 0: horizontal, 1: vertical

# --- Predicates ---

@dataclass
class ColorMatch(Predicate):
    color: int

@dataclass
class AreaRange(Predicate):
    min_area: int
    max_area: int

@dataclass
class ShapeMatch(Predicate):
    shape_type: str # 'rectangle', 'square', 'line'

@dataclass
class LogicOp(Predicate):
    op: str # 'AND', 'OR', 'NOT'
    left: Predicate
    right: Optional[Predicate] = None

# --- Tier 3: Generative Operations ---

@dataclass
class FloodFill(Action):
    start_x: int
    start_y: int
    color: int
    boundary_color: int

@dataclass
class DrawLine(Action):
    start_x: int
    start_y: int
    end_x: int
    end_y: int
    color: int

@dataclass
class DrawRect(Action):
    x: int
    y: int
    width: int
    height: int
    color: int
    filled: bool

@dataclass
class ExtrudePattern(Action):
    pattern_id: int
    direction: int
    count: int

@dataclass
class MirrorAcross(Action):
    axis_type: int # 0: vertical, 1: horizontal
    axis_pos: int

@dataclass
class CompleteSymmetry(Action):
    symmetry_type: int # 0: horizontal, 1: vertical, 2: diagonal

@dataclass
class ConnectObjects(Action):
    obj_a: int
    obj_b: int
    style: int # 0: line, 1: L-shape

@dataclass
class Crop(Action):
    x: int
    y: int
    width: int
    height: int

@dataclass
class Paste(Action):
    subgrid_id: int
    x: int
    y: int

@dataclass
class Recolor(Action):
    object_id: int
    new_color: int
