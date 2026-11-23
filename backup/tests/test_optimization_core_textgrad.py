"""Tests for optimization core + textgrad_lib skeleton."""

from __future__ import annotations

import pytest

from agent_framework.optimization.core.rollback import should_accept_candidate


def test_should_accept_candidate_with_rollback():
    assert should_accept_candidate(0.9, 0.8, rollback=True)
    assert not should_accept_candidate(0.7, 0.8, rollback=True)


def test_should_accept_candidate_without_rollback():
    assert should_accept_candidate(0.1, 0.9, rollback=False)


def test_require_textgrad_imports_when_installed():
    pytest.importorskip("textgrad")
    from agent_framework.optimization.optimizers.textgrad_lib._import import require_textgrad

    tg, Variable, TextLoss, TextualGradientDescent = require_textgrad()
    assert tg is not None
    assert Variable is not None
    assert TextLoss is not None
    assert TextualGradientDescent is not None


@pytest.mark.textgrad
def test_decomposition_prompt_variable_keeps_placeholders():
    textgrad = pytest.importorskip("textgrad")
    assert textgrad is not None

    from agent_framework.optimization.optimizers.textgrad_lib.adapter import (
        decomposition_prompt_variable,
        read_prompt_value,
    )

    prompt = (
        "背景: {background_info}\n"
        "团队: {agent_team}\n"
        "输入: {user_input}"
    )
    var = decomposition_prompt_variable(prompt)
    assert read_prompt_value(var) == prompt
