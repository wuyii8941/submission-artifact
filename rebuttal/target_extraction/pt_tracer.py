import torch

kernels = []
kernel_names = []
_current = []
_installed = False


def install():
    global _installed
    if _installed:
        return
    from torch._inductor.codegen.triton import TritonKernel
    from torch._inductor.codegen.triton_split_scan import TritonSplitScanKernel

    orig_codegen = TritonKernel.codegen_kernel

    def codegen_kernel(self, *args, **kwargs):
        features = getattr(self, 'features', None)
        if isinstance(self, TritonSplitScanKernel):
            family = 'split_scan'
        elif getattr(self, 'cooperative_reduction', False):
            family = 'cooperative_reduction'
        elif features is not None and features.contains_op('scan'):
            family = 'scan'
        elif getattr(self, 'persistent_reduction', False):
            family = 'persistent_reduction'
        elif getattr(self, 'inside_reduction', False):
            family = 'reduction'
        else:
            family = 'pointwise'
        rec = {'family': family, 'heuristic': None}
        _current.append(rec)
        try:
            return orig_codegen(self, *args, **kwargs)
        finally:
            _current.pop()
            kernels.append(rec)
    TritonKernel.codegen_kernel = codegen_kernel

    for cls in (TritonKernel, TritonSplitScanKernel):
        orig = cls.__dict__['_get_heuristic']

        def wrapped(self, _orig=orig):
            value = _orig(self)
            if _current:
                _current[-1]['heuristic'] = value
            return value
        cls._get_heuristic = wrapped

    try:
        from torch._inductor.codegen.wrapper import PythonWrapperCodegen as Wrapper
    except ImportError:
        from torch._inductor.codegen.wrapper import WrapperCodeGen as Wrapper
    orig_call = Wrapper.generate_kernel_call
    orig_out = Wrapper.generate_extern_kernel_out
    orig_alloc = Wrapper.generate_extern_kernel_alloc

    def call(self, kernel_name, *args, **kwargs):
        kernel_names.append(str(kernel_name))
        return orig_call(self, kernel_name, *args, **kwargs)

    def out(self, node, *args, **kwargs):
        kernel_names.append(f'extern_out:{str(node)[:200]}')
        return orig_out(self, node, *args, **kwargs)

    def alloc(self, node, *args, **kwargs):
        kernel_names.append(f'extern_alloc:{str(node)[:200]}')
        return orig_alloc(self, node, *args, **kwargs)
    Wrapper.generate_kernel_call, Wrapper.generate_extern_kernel_out, Wrapper.generate_extern_kernel_alloc = call, out, alloc
    _installed = True


def reset():
    kernels.clear()
    kernel_names.clear()


def observed():
    kinds = {k['family'] for k in kernels}
    for name in kernel_names:
        if name.startswith('extern'):
            low = name.lower()
            kinds.add('extern_gemm' if any(t in low for t in ('mm', 'addmm', 'bmm', 'linear')) else
                      'extern_conv' if 'conv' in low else 'aten_fallback')
    agree = all(k['heuristic'] == k['family'] or (k['family'] == 'scan' and k['heuristic'] in
                ('persistent_reduction', 'reduction')) for k in kernels)
    return sorted(kinds), agree
