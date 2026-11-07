"""Read device/compiler support independently of trial success counts."""
import hashlib,inspect,json
from pathlib import Path
from inventory import ROOT
from common_domain import descriptor
import torch
import torch._inductor.compile_fx as compile_fx
import torch._inductor.config as config


def main():
    out=ROOT/'device_domain_audit_v1';out.mkdir(exist_ok=False)
    props=torch.cuda.get_device_properties(0);check=compile_fx._check_triton_bf16_support
    path=Path(inspect.getsourcefile(check));lines,line=inspect.getsourcelines(check)
    result={'control_only':True,'torch':torch.__version__,'commit':torch.version.git_version,
            'device':props.name,'capability':[props.major,props.minor],
            'bf16_supported_native':torch.cuda.is_bf16_supported(including_emulation=False),
            'bf16_supported_including_emulation':torch.cuda.is_bf16_supported(including_emulation=True),
            'compiler_support_guard':{'source':str(path),'line':line,'text':''.join(lines),'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest()},
            'common_eager_semantic_domain':descriptor(),'coordinate_descent_tuning':config.coordinate_descent_tuning,
            'classification':'BF16 compile support is separate from eager legality; retain requests and costs, do not treat support rejection as a discovered bug',
            'no_dtype_removed_from_generator_or_primary_denominator_by_this_audit':True,
            'primary_denominator_committed':False,'effect_experiment_started':False}
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({key:result[key] for key in ('device','capability','bf16_supported_native','bf16_supported_including_emulation','coordinate_descent_tuning')}))
if __name__=='__main__':main()
