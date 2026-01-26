```py
@dataclass(frozen=True)
class AttackInput:
    malicious_task: MaliciousTask
    injectable_model_request: InjectableModelRequest
    message_history: list[ModelMessage]
    previous_attempts: list[AttackAttempt]
```

```py
class AbstractAttack(Protocol):
    def compute_injection(
        self,
        attack_input: AttackInput,
    ): ...

    def compute_condition(self, malicious_task: MaliciousTask) -> bool: ...
```

```py
class TemplateAttack(AbstractAttack):
  def def compute_injection(
      self,
      attack_input: AttackInput,
  ):
      ...
```
