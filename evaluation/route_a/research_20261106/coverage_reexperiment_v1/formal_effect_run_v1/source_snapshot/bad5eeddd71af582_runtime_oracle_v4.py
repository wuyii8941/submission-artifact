"""Shared V4 eager-vs-compiled output oracle.

The oracle is intentionally independent of a compiler decision and is usable
by every future arm.  It rejects non-finite floating references instead of
silently treating them as a correctness pass.
"""
import torch


def compare(reference, actual, *, atol=1e-5, rtol=1e-4):
    if isinstance(reference, (tuple, list)):
        if type(reference) is not type(actual) or len(reference) != len(actual):
            return {'kind': 'sequence', 'status': 'candidate', 'reason': 'structure_mismatch'}
        children = [compare(left, right, atol=atol, rtol=rtol) for left, right in zip(reference, actual)]
        statuses = {item['status'] for item in children}
        status = ('candidate' if 'candidate' in statuses else
                  'oracle_inapplicable_nonfinite_reference' if 'oracle_inapplicable_nonfinite_reference' in statuses else 'pass')
        return {'kind': 'sequence', 'status': status, 'children': children}
    if isinstance(reference, dict):
        if not isinstance(actual, dict) or reference.keys() != actual.keys():
            return {'kind': 'mapping', 'status': 'candidate', 'reason': 'structure_mismatch'}
        children = {str(key): compare(reference[key], actual[key], atol=atol, rtol=rtol) for key in reference}
        statuses = {item['status'] for item in children.values()}
        status = ('candidate' if 'candidate' in statuses else
                  'oracle_inapplicable_nonfinite_reference' if 'oracle_inapplicable_nonfinite_reference' in statuses else 'pass')
        return {'kind': 'mapping', 'status': status, 'children': children}
    if isinstance(reference, torch.Tensor) != isinstance(actual, torch.Tensor):
        return {'kind': 'type_mismatch', 'status': 'candidate'}
    if isinstance(reference, torch.Tensor) and isinstance(actual, torch.Tensor):
        result = {'kind': 'tensor', 'reference_shape': list(reference.shape),
                  'actual_shape': list(actual.shape), 'reference_dtype': str(reference.dtype),
                  'actual_dtype': str(actual.dtype), 'numel': reference.numel()}
        if reference.shape != actual.shape or reference.dtype != actual.dtype:
            result['status'] = 'candidate'; result['reason'] = 'shape_or_dtype_mismatch'
            return result
        if reference.dtype.is_floating_point or reference.dtype.is_complex:
            finite = bool(torch.isfinite(reference).all())
            result['reference_finite'] = finite
            result['compiled_finite'] = bool(torch.isfinite(actual).all())
            if not finite:
                result['status'] = 'oracle_inapplicable_nonfinite_reference'
                return result
            result.update(atol=atol, rtol=rtol,
                          status='pass' if bool(torch.allclose(reference, actual, atol=atol, rtol=rtol)) else 'candidate')
            return result
        result['status'] = 'pass' if bool(torch.equal(reference, actual)) else 'candidate'
        return result
    result = {'kind': 'scalar_or_other', 'reference_type': type(reference).__name__,
              'actual_type': type(actual).__name__, 'status': 'pass' if reference == actual else 'candidate'}
    return result
