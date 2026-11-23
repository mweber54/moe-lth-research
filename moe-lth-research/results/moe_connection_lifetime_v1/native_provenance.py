"""Record loaded module, native configs and archive-oracle comparison."""
import inspect
import json
import subprocess
import sys

from groups import HERE,V2,branch
from moe_lth.models.experts import ExpertFFN
from moe_lth.models.moe_layer import MoEFeedForward
from moe_lth.models.router import TopKRouter
from stage import records
from pathlib import Path


def main():
    reference="95d2543fb213246d07c47919157b3180be9325f2"
    import hashlib
    parts={}
    for cls in (ExpertFFN,MoEFeedForward,TopKRouter,branch.TinyMoELanguageModel):
        path=Path(inspect.getsourcefile(cls))
        relative=path.relative_to(HERE.parents[1])
        ref=subprocess.check_output(["git","show",f"{reference}:./{relative}"],cwd=HERE.parents[1])
        parts[cls.__name__]=dict(path=str(path),loaded_sha256=branch.sha(path),reference_sha256=hashlib.sha256(ref).hexdigest())
    known=json.loads((V2/"input_manifest.json").read_text())["cells"]
    native=json.loads((HERE.parents[1]/"results/imp_recovery_order_stability_v1/canonical_stream_manifest.json").read_text())["seeds"]["7"]["config"]
    branch.save_json(HERE/"native_provenance.json",dict(runtime=sys.executable,torch=branch.torch.__version__,
        model=parts,reference=reference,source=records(),config=native,
        archived_inputs={cell:entry["files"] for cell,entry in known.items()},
        graph="biased linear, exact GELU, inverted dropout (p=0 natively), biased linear; selected probability scales accepted output externally",
        reference_comparison="The expert forward is byte-identical. The loaded MoE and transformer support optional routing overrides, capacity controls and chunking. The study calls the default learned dispatch with capacity enabled.",
        recipients="final-layer fc1.weight rows from exclusive fc2 columns; fc1.bias excluded",
        optimizer="native torch.optim.AdamW, AMP FP16 GradScaler, global clip 1.0, lr 0.0003, decay 0.01, no scheduler",
        roster_evidence={"v1":json.loads((branch.V1/"registration.json").read_text()),"v2":json.loads((V2/"registration.json").read_text())}))
if __name__=="__main__":
    main()
