from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


class ExprNode:
    pass


@dataclass(frozen=True, slots=True)
class NumberNode(ExprNode):
    value: float


@dataclass(frozen=True, slots=True)
class IdentifierNode(ExprNode):
    name: str


@dataclass(frozen=True, slots=True)
class UnaryOpNode(ExprNode):
    operator: str
    operand: ExprNode


@dataclass(frozen=True, slots=True)
class BinaryOpNode(ExprNode):
    operator: str
    left: ExprNode
    right: ExprNode


@dataclass(frozen=True, slots=True)
class FunctionCallNode(ExprNode):
    name: str
    args: tuple[ExprNode, ...]


def iter_child_nodes(node: ExprNode) -> Iterable[ExprNode]:
    if isinstance(node, UnaryOpNode):
        yield node.operand
    elif isinstance(node, BinaryOpNode):
        yield node.left
        yield node.right
    elif isinstance(node, FunctionCallNode):
        yield from node.args


def node_depth(node: ExprNode) -> int:
    children = list(iter_child_nodes(node))
    if not children:
        return 1
    return 1 + max(node_depth(child) for child in children)


def node_complexity(node: ExprNode) -> int:
    return 1 + sum(node_complexity(child) for child in iter_child_nodes(node))


def to_expression(node: ExprNode) -> str:
    if isinstance(node, NumberNode):
        return str(int(node.value)) if float(node.value).is_integer() else repr(node.value)
    if isinstance(node, IdentifierNode):
        return node.name
    if isinstance(node, UnaryOpNode):
        return f"({node.operator}{to_expression(node.operand)})"
    if isinstance(node, BinaryOpNode):
        return f"({to_expression(node.left)}{node.operator}{to_expression(node.right)})"
    if isinstance(node, FunctionCallNode):
        return f"{node.name}({','.join(to_expression(arg) for arg in node.args)})"
    raise TypeError(f"Unsupported node type: {type(node)!r}")
