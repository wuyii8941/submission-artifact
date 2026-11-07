"""Flag-gated facade; the legacy request, adapter and rendered bytes stay frozen."""
from dataclasses import dataclass
from typing import Optional
from .model import GraphRequest
from .torchfuzz_adapter import TorchFuzzAdapter
from .conditions_v2 import ConditionSpec, generate_case


@dataclass(frozen=True)
class ConfiguredGraphRequest(GraphRequest):
    condition_system: str = 'legacy'
    condition_v2: Optional[ConditionSpec] = None

    def to_dict(self):
        data = super().to_dict()
        if self.condition_system != 'legacy' or self.condition_v2 is not None:
            data.update(condition_system=self.condition_system,
                        condition_v2=self.condition_v2.to_dict() if self.condition_v2 else None)
        return data


class ConfiguredAdapter(TorchFuzzAdapter):
    def generate(self, request):
        mode = getattr(request, 'condition_system', 'legacy')
        if mode == 'v2':
            return generate_case(self, request)
        if mode != 'legacy' or getattr(request, 'condition_v2', None) is not None:
            raise ValueError('condition_v2_requires_explicit_v2_flag')
        return super().generate(request)
