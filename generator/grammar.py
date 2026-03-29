from __future__ import annotations

import random


class GrammarExpressionGenerator:
    def __init__(
        self,
        fields: list[str],
        lookbacks: list[int],
        max_depth: int,
        seed: int,
    ) -> None:
        self.fields = fields
        self.lookbacks = lookbacks
        self.max_depth = max_depth
        self.random = random.Random(seed)
        self.unary_wrappers = ["rank", "zscore", "sign", "abs"]
        self.window_operators = [
            "ts_delay",
            "ts_delta",
            "ts_mean",
            "ts_std_dev",
            "ts_min",
            "ts_max",
            "ts_decay_linear",
            "ts_rank",
            "ts_sum",
        ]
        self.binary_window_operators = ["ts_corr", "ts_covariance"]
        self.arithmetic_operators = ["+", "-", "*", "/"]

    def generate(self, count: int) -> list[str]:
        expressions: list[str] = []
        while len(expressions) < count:
            expressions.append(self._generate_expr(depth=1))
        return expressions

    def _generate_expr(self, depth: int) -> str:
        if depth >= self.max_depth or self.random.random() < 0.25:
            return self.random.choice(self.fields)

        branch = self.random.choices(
            population=["unary", "window", "binary_window", "arithmetic", "log_branch"],
            weights=[0.20, 0.30, 0.15, 0.25, 0.10],
            k=1,
        )[0]

        if branch == "unary":
            operator = self.random.choice(self.unary_wrappers)
            return f"{operator}({self._generate_expr(depth + 1)})"
        if branch == "window":
            operator = self.random.choice(self.window_operators)
            lookback = self.random.choice(self.lookbacks)
            return f"{operator}({self._generate_expr(depth + 1)}, {lookback})"
        if branch == "binary_window":
            operator = self.random.choice(self.binary_window_operators)
            lookback = self.random.choice(self.lookbacks)
            return (
                f"{operator}({self._generate_expr(depth + 1)}, "
                f"{self._generate_expr(depth + 1)}, {lookback})"
            )
        if branch == "log_branch":
            return f"log((abs({self._generate_expr(depth + 1)}) + 1))"
        left = self._generate_expr(depth + 1)
        right = self._generate_expr(depth + 1)
        operator = self.random.choice(self.arithmetic_operators)
        return f"({left} {operator} {right})"
