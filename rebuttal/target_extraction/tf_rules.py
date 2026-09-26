def signature_class(od):
    ins, outs = od.input_arg, od.output_arg
    attrs = {a.name for a in od.attr}
    in_names = [a.name for a in ins]
    if 'reduction_indices' in in_names or 'dimension' in in_names or ('axis' in in_names and 'keep_dims' in attrs):
        return 'reduction'
    if od.name.startswith('Cum'):
        return 'scan'
    if 'MatMul' in od.name or od.name.startswith('Einsum'):
        return 'linalg'
    if od.name.startswith('Conv') or 'Pool' in od.name:
        return 'conv_pool'
    if len(ins) == 1 and len(outs) == 1 and ins[0].type_attr and ins[0].type_attr == outs[0].type_attr and not ins[0].number_attr:
        return 'elementwise_unary'
    if len(ins) == 2 and len(outs) == 1 and ins[0].type_attr and ins[0].type_attr == ins[1].type_attr == outs[0].type_attr:
        return 'elementwise_binary'
    return 'other'
