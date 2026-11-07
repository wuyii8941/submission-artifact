"""Generic value-domain screen for undefined integer arithmetic.

This is an operator semantic rule, shared by every arm. It observes an eager
execution and rejects a concrete invalid value combination without changing
inputs, operator pool, topology, or the predeclared source denominator.
Nonfinite floating results remain separately labelled by the numerical oracle.
"""
import torch
from torch.utils._python_dispatch import TorchDispatchMode


class ValueDomainViolation(ValueError):
    def __init__(self, evidence):
        super().__init__(evidence['reason'])
        self.evidence = evidence


def any_true(value):
    return bool(value.any().item()) if isinstance(value, torch.Tensor) else bool(value)


def result_dtype(left, right):
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return torch.result_type(left, right)
    return torch.tensor(left + right).dtype


class ValueDomain(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.checks = []; self.calls = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}; name = func._schema.name
        self.calls += 1
        integer_remainder = name in ('aten::fmod', 'aten::remainder')
        rounded_division = name == 'aten::div' and kwargs.get('rounding_mode') in ('floor', 'trunc')
        integer_power = name == 'aten::pow'
        if len(args) >= 2 and (integer_remainder or rounded_division or integer_power):
            left, right = args[:2]; dtype = result_dtype(left, right)
            if not (dtype.is_floating_point or dtype.is_complex):
                evidence = {'aten': str(func), 'schema': str(func._schema), 'result_dtype': str(dtype)}
                if integer_remainder or rounded_division:
                    zero = any_true(right == 0)
                    evidence['integer_divisor_nonzero'] = not zero
                    if zero:
                        evidence['reason'] = 'integer_zero_divisor'
                        self.checks.append(evidence); raise ValueDomainViolation(evidence)
                    if dtype != torch.bool:
                        minimum = torch.iinfo(dtype).min
                        # Signed minimum / -1 is overflow-sensitive. Preserve
                        # this as a rejected value combination, not a repair.
                        overflow = any_true((left == minimum) & (right == -1))
                        evidence['signed_division_overflow_absent'] = not overflow
                        if overflow:
                            evidence['reason'] = 'signed_integer_division_overflow'
                            self.checks.append(evidence); raise ValueDomainViolation(evidence)
                if integer_power:
                    negative = any_true(right < 0)
                    evidence['integer_exponent_nonnegative'] = not negative
                    if negative:
                        evidence['reason'] = 'negative_integer_exponent'
                        self.checks.append(evidence); raise ValueDomainViolation(evidence)
                evidence['status'] = 'accepted_by_checked_value_rules'
                self.checks.append(evidence)
        return func(*args, **kwargs)

    def receipt(self):
        return {'schema': 'common-value-domain-screen-v1', 'dispatch_calls': self.calls, 'checks': self.checks,
                'scope': 'integer divisor, signed division overflow and integer exponent rules only',
                'complete_value_domain_proven': False, 'mutates_inputs': False}
