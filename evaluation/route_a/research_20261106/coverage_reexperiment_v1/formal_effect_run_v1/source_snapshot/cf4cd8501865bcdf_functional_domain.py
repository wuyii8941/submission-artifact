"""Enforce the common forward, functional emission domain in all arms."""
import ast


def validate_functional_source(source):
    tree=ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node,(ast.AugAssign,ast.Delete,ast.Yield,ast.YieldFrom,ast.Await)):
            raise ValueError('outside_functional_forward_domain:'+type(node).__name__)
        if isinstance(node,ast.Assign) and any(not isinstance(target,ast.Name) for target in node.targets):
            raise ValueError('functional_domain_requires_name_assignment')
        if not isinstance(node,ast.Call):continue
        for kw in node.keywords:
            if kw.arg is None:raise ValueError('functional_domain_disallows_dynamic_kwargs')
            if kw.arg=='out':raise ValueError('functional_domain_disallows_out')
            if kw.arg=='inplace' and not (isinstance(kw.value,ast.Constant) and kw.value.value is False):
                raise ValueError('functional_domain_disallows_inplace')
        if isinstance(node.func,ast.Attribute) and node.func.attr.endswith('_') and not node.func.attr.startswith('__'):
            raise ValueError('functional_domain_disallows_mutating_method')
        if isinstance(node.func,ast.Attribute) and node.func.attr in ('backward','grad','requires_grad_'):
            raise ValueError('functional_domain_disallows_backward')
    return True
