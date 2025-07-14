from collections import defaultdict
from dataclasses import field
from typing import List, Dict, Self, Set, FrozenSet, Tuple, Literal, Optional, Any, Mapping

from functools import lru_cache

from .types_base import PType, AnyPType, OperatorPrecedence
from .util import fnr_dataclass, union_dict


@fnr_dataclass
class BaseAPIObject(PType):
    """
    Base class for all API values
    """

    @property
    def root_values(self) -> Set[Self]:
        return {self}

    def __ge__(self, other):
        return isinstance(other, BaseAPIObject)

    def __eq__(self, other):
        return type(other) is type(self)

    def __hash__(self):
        return hash(self.__class__)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "any"

    def instantiate_type_params(self, type_params: Dict[str, Self]) -> Self:
        return self


@fnr_dataclass
class PrimitivePType(BaseAPIObject):
    """
    Primitive type
    """

    @property
    def root_values(self) -> Set[Self]:
        return {self}

    def __ge__(self, other):
        return issubclass(other.__class__, self.__class__)

    def __eq__(self, other):
        return self.__class__ == other.__class__

    def __hash__(self):
        return hash(self.__class__)

    def instantiate_type_params(self, type_params: Dict[str, Self]) -> Self:
        return self


@fnr_dataclass
class StringPType(PrimitivePType):
    """
    String type
    """
    def __str__(self):
        return "string"


@fnr_dataclass
class FloatPType(PrimitivePType):
    """
    Float type
    """
    def __str__(self):
        return "float"


@fnr_dataclass
class IntegerPType(PrimitivePType):
    """
    Integer type
    """
    def __str__(self):
        return "integer"


@fnr_dataclass
class BooleanPType(PrimitivePType):
    """
    Boolean type
    """
    def __str__(self):
        return "boolean"


@fnr_dataclass
class TimestampPType(PrimitivePType):
    """
    Timestamp type
    """
    def __str__(self):
        return "timestamp"


@fnr_dataclass
class NullPType(PrimitivePType):
    """
    Null type (for explicit null values)
    """
    def __str__(self):
        return "null"

@fnr_dataclass
class BlobPType(PrimitivePType):
    """
    Binary/blob type (for e.g. S3 Body)
    """
    def __str__(self):
        return "blob"

@fnr_dataclass
class EnumPType(PrimitivePType):
    """
    Enum type (for fields with a fixed set of string values)
    """
    values: List[str]
    def __str__(self):
        return f"enum({', '.join(self.values)})"

@fnr_dataclass
class UnionPType(BaseAPIObject):
    """
    Union type (for fields that can be one of several types)
    """
    types: List[PType]
    def __str__(self):
        return f"({' | '.join(str(t) for t in self.types)})"

@fnr_dataclass
class ArrayPType(BaseAPIObject):
    """
    Array type (for lists of elements)
    """
    element_type: PType
    def __str__(self):
        return f"{self.element_type}[]"

@fnr_dataclass
class MapPType(BaseAPIObject):
    """
    Map type (parameterized by key and value type)
    """
    key_type: PType
    value_type: PType
    def __str__(self):
        return f"map<{self.key_type}, {self.value_type}>"

@fnr_dataclass
class StructurePType(BaseAPIObject):
    """
    Structure type (JSON object with named fields, e.g. AWS API response)
    fields: Dict[str, PType] - field name to type
    required_fields: Set[str] - required field names
    field_metadata: Optional[Dict[str, Dict[str, Any]]] - optional metadata per field (e.g. description)
    """
    fields: Dict[str, PType]
    required_fields: Set[str] = field(default_factory=set)
    field_metadata: Optional[Dict[str, Dict[str, Any]]] = None
    def __str__(self):
        req = ', '.join(f for f in self.fields if f in self.required_fields)
        opt = ', '.join(f for f in self.fields if f not in self.required_fields)
        return f"structure(required=[{req}], optional=[{opt}])"

def any_reachable(
    typs: Set[PType],
    goal_t: PType,
    min_operator_prec: OperatorPrecedence,
    max_operator_prec: OperatorPrecedence,
    in_array: List[OperatorPrecedence],
    in_nested_expression: List[OperatorPrecedence],
    in_pattern: List[Tuple[PType, PType, OperatorPrecedence]],
    max_steps=5,
    max_depth=(0, 0),
):
    return any(
        reachable(
            t,
            goal_t,
            min_operator_prec,
            max_operator_prec,
            in_array,
            in_nested_expression,
            in_pattern,
            max_steps,
            max_depth,
        )
        for t in typs
    )


# TODO: do I need this?
# TODO: remove nesting depth? Or add it to the classes above?
def reachable(
    t: PType,
    goal_t: PType,
    min_operator_prec: OperatorPrecedence,
    max_operator_prec: OperatorPrecedence,
    in_array: List[OperatorPrecedence],
    in_nested_expression: List[OperatorPrecedence],
    in_pattern: List[Tuple[PType, PType, OperatorPrecedence]],
    max_steps=5,
    max_depth=(0, 0),
):
    max_depth = tuple(
        max(xs)
        for xs in zip(
            max_depth,
            t.nesting_depth,
            goal_t.nesting_depth,
            *[p[0].nesting_depth for p in in_pattern],
        )
    )
    res = _reachable_bfs(
        t,
        goal_t,
        min_operator_prec,
        max_operator_prec,
        tuple(in_array),
        tuple(in_nested_expression),
        tuple(in_pattern),
        max_depth,
        max_steps=max_steps,
    )
    # print(t, goal_t, res)
    return res == "REACHABLE"
