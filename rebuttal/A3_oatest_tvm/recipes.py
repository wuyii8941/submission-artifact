import tvm
from tvm import relax

PIPELINE = (
    ("LegalizeOps", lambda: relax.transform.LegalizeOps()),
    ("AnnotateTIROpPattern", lambda: relax.transform.AnnotateTIROpPattern()),
    ("FoldConstant", lambda: relax.transform.FoldConstant()),
    ("FuseOps", lambda: relax.transform.FuseOps()),
    ("FuseTIR", lambda: relax.transform.FuseTIR()),
    ("DispatchSampling", lambda: relax.backend.DispatchSampling()),
    ("DispatchSortScan", lambda: relax.backend.DispatchSortScan()),
    ("LegalizeOps#2", lambda: relax.transform.LegalizeOps()),
    ("RewriteDataflowReshape", lambda: relax.transform.RewriteDataflowReshape()),
    ("ToNonDataflow", lambda: relax.transform.ToNonDataflow()),
    ("RemovePurityChecking", lambda: relax.transform.RemovePurityChecking()),
    ("CallTIRRewrite", lambda: relax.transform.CallTIRRewrite()),
    ("StaticPlanBlockMemory", lambda: relax.transform.StaticPlanBlockMemory()),
    ("RewriteCUDAGraph", lambda: relax.transform.RewriteCUDAGraph()),
    ("LowerAllocTensor", lambda: relax.transform.LowerAllocTensor()),
    ("KillAfterLastUse", lambda: relax.transform.KillAfterLastUse()),
    ("VMBuiltinLower", lambda: relax.transform.VMBuiltinLower()),
    ("ComputePrimValue", lambda: relax.transform.ComputePrimValue()),
    ("VMShapeLower", lambda: relax.transform.VMShapeLower()),
    ("AttachGlobalSymbol", lambda: relax.transform.AttachGlobalSymbol()),
)
PIPELINE_NAMES = tuple(name for name, _ in PIPELINE)

TARGETS = (
    "AdjustMatmulOrder", "AllocateWorkspace", "AnnotateTIROpPattern",
    "AttachGlobalSymbol", "BundleModelParams", "CanonicalizeBindings",
    "CombineParallelMatmul", "ComputePrimValue", "ConvertToDataflow",
    "DeadCodeElimination", "DecomposeOpsForInference", "DecomposeOpsForTraining",
    "EliminateCommonSubexpr", "ExpandMatmulOfSum", "ExpandTupleArguments",
    "FewShotTuning", "FoldConstant", "FuseOps", "FuseTIR",
    "InlinePrivateFunctions", "KillAfterLastUse", "LambdaLift", "LazyGetInput",
    "LazySetOutput", "LegalizeOps", "LiftTransformParams", "LowerAllocTensor",
    "MergeCompositeFunctions", "Normalize", "RealizeVDevice",
    "RemovePurityChecking", "RemoveUnusedOutputs", "RemoveUnusedParameters",
    "ReorderPermuteDimsAfterConcat", "ReorderTakeAfterMatmul", "RewriteCUDAGraph",
    "RewriteDataflowReshape", "StaticPlanBlockMemory", "ToMixedPrecision",
    "TopologicalSort", "VMBuiltinLower",
)

PASS_CONFIG = {"RewriteCUDAGraph": {"relax.backend.use_cuda_graph": True}}

INFO_NAME = {"DecomposeOpsForInference": "DecomposeOps",
             "DecomposeOpsForTraining": "DecomposeOps",
             "ConvertToDataflow": "ConvertToDataflow"}
EVENT_TO_TARGETS = {}
for _target in TARGETS:
    EVENT_TO_TARGETS.setdefault(INFO_NAME.get(_target, _target), []).append(_target)


def make_pass(name):
    return getattr(relax.transform, name)()


def prefix_of(target):
    if target not in PIPELINE_NAMES or PIPELINE_NAMES.index(target) == 0:
        return None
    return PIPELINE[:PIPELINE_NAMES.index(target)]


def context(target, instruments=()):
    config = PASS_CONFIG.get(target, {})
    return tvm.transform.PassContext(opt_level=3, config=config,
                                     instruments=list(instruments))
