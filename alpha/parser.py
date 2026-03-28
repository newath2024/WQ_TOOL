from __future__ import annotations

import ast

from alpha.ast_nodes import BinaryOpNode, ExprNode, FunctionCallNode, IdentifierNode, NumberNode, UnaryOpNode


UNARY_OPERATORS = {
    ast.UAdd: "+",
    ast.USub: "-",
}

BINARY_OPERATORS = {
    ast.Add: "+",
    ast.Sub: "-",
    ast.Mult: "*",
    ast.Div: "/",
}


class ExpressionParser:
    def parse(self, expression: str) -> ExprNode:
        try:
            parsed = ast.parse(expression, mode="eval")
        except SyntaxError as exc:
            raise ValueError(f"Invalid expression syntax: {expression}") from exc
        return self._convert(parsed.body)

    def _convert(self, node: ast.AST) -> ExprNode:
        if isinstance(node, ast.Name):
            return IdentifierNode(name=node.id)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return NumberNode(value=float(node.value))
        if isinstance(node, ast.UnaryOp) and type(node.op) in UNARY_OPERATORS:
            return UnaryOpNode(operator=UNARY_OPERATORS[type(node.op)], operand=self._convert(node.operand))
        if isinstance(node, ast.BinOp) and type(node.op) in BINARY_OPERATORS:
            return BinaryOpNode(
                operator=BINARY_OPERATORS[type(node.op)],
                left=self._convert(node.left),
                right=self._convert(node.right),
            )
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
            return FunctionCallNode(
                name=node.func.id,
                args=tuple(self._convert(argument) for argument in node.args),
            )
        raise ValueError(f"Unsupported expression construct: {ast.dump(node)}")


def parse_expression(expression: str) -> ExprNode:
    return ExpressionParser().parse(expression)
