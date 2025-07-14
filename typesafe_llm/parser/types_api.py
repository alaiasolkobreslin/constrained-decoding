from collections import defaultdict
from dataclasses import field
from typing import List, Dict, Self, Set, FrozenSet, Tuple, Literal, Optional

from functools import lru_cache

from .types_base import PType, AnyPType, OperatorPrecedence
from .util import fnr_dataclass, union_dict


@fnr_dataclass
class BaseAPIObject(PType):
    """
    Base class for all API values
    """

    @property
    def _attributes(self) -> Dict[str, Self]:
        """
        Get all attributes of the object
        """
        return {}

    @property
    def attributes(self) -> Dict[str, Tuple[Self, bool]]:
        return {
            k: (v, isinstance(v, FunctionPType)) for k, v in self._attributes.items()
        }

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

    @property
    def _attributes(self) -> Dict[str, Self]:
        res = {}
        res.update(super()._attributes)
        return res

    def __str__(self):
        return "string"


@fnr_dataclass
class FloatPType(PrimitivePType):
    """
    Float type
    """

    @property
    def _attributes(self) -> Dict[str, Self]:
        res = {}
        res.update(super()._attributes)
        return res

    def __str__(self):
        return "float"


@fnr_dataclass
class IntegerPType(PrimitivePType):
    """
    Integer type
    """

    @property
    def _attributes(self) -> Dict[str, Self]:
        res = {}
        res.update(super()._attributes)
        return res

    def __str__(self):
        return "integer"


@fnr_dataclass
class BooleanPType(PrimitivePType):
    """
    Boolean type
    """

    @property
    def _attributes(self) -> Dict[str, Self]:
        res = {}
        res.update(super()._attributes)
        return res

    def __str__(self):
        return "boolean"


@fnr_dataclass
class TimestampPType(PrimitivePType):
    """
    Timestamp type
    """

    @property
    def _attributes(self) -> Dict[str, Self]:
        res = {}
        res.update(super()._attributes)
        return res

    def __str__(self):
        return "timestamp"


@fnr_dataclass
class MapPType(BaseAPIObject):
    """
    Map type
    """

    @property
    def _attributes(self) -> Dict[str, Self]:
        res = {}
        res.update(super()._attributes)
        return res

    def __str__(self):
        return "map"

@fnr_dataclass
class StructurePType(BaseAPIObject):  # JSON object
    """
    Structure type
    """

    @property
    def _attributes(self) -> Dict[str, Self]:
        res = {}
        res.update(super()._attributes)
        return res

    def __str__(self):
        return "structure"

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
