from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import logfire
from jinja2 import Environment, StrictUndefined, TemplateSyntaxError
from prompt_siren.agents import AbstractAgent
from prompt_siren.attacks.template_string_attack import _get_model_name
from pydantic_ai.messages import ModelMessage, ModelResponse

from ..tasks import MaliciousTask
from ..types import (
    InjectableModelRequest,
    InjectableModelRequestPart,
    InjectionAttack,
    InjectionAttacksDict,
    StrContentAttack,
)


class AttackAttempt:
    injections: InjectionAttack
    response: ModelResponse
    message_history: list[ModelMessage]


@dataclass(frozen=True)
class AttackInput:
    malicious_task: MaliciousTask
    injectable_model_request: InjectableModelRequest
    message_history: list[ModelMessage]
    previous_attempts: list[AttackAttempt]


class AbstractAttack(Protocol):
    def compute_injection(
        self,
        agent: AbstractAgent,
        attack_input: AttackInput,
    ) -> InjectionAttacksDict[StrContentAttack]: ...

    def should_stop(
        self, malicious_task: MaliciousTask, previous_attempt: AttackAttempt
    ) -> bool: ...


class BatchAbstractAttack(Protocol):
    def compute_batch_injection(
        self,
        agent: AbstractAgent,
        attack_inputs: list[AttackInput],
    ) -> InjectionAttacksDict[StrContentAttack]: ...

    def should_stop(self, malicious_tasks: list[MaliciousTask]) -> bool: ...


class TemplateAttack(AbstractAttack):
    def compute_injection(
        self,
        agent: AbstractAgent,
        attack_input: AttackInput,
    ) -> InjectionAttacksDict[StrContentAttack]:
        attacks = {}
        for vector_id in attack_input.injectable_model_request_part.vector_ids:
            # Create Jinja2 environment and render template
            env = Environment(undefined=StrictUndefined)
            template = env.from_string(self.config.attack_template)

            # Merge default fields with custom template fields
            template_context = {
                "goal": attack_input.malicious_task.goal,
                **self.config.template_fields,
            }

            # Add model name if not already provided in template_fields
            if "model" not in template_context:
                template_context["model"] = _get_model_name(agent.get_agent_name())

            try:
                injection = template.render(**template_context)
            except TemplateSyntaxError as e:
                logfire.error(
                    "Jinja2 template syntax error",
                    error=str(e),
                    template=self.config.attack_template,
                )
                raise

            logfire.info(
                "creating injection",
                injection=injection,
                template_short_name=self.config.template_short_name,
            )
            attacks[vector_id] = StrContentAttack(content=injection)

        return attacks
