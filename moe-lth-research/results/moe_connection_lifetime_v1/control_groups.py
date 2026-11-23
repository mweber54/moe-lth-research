from groups import HERE,V2,branch,construct,json,torch

cell="seed7_A"
folder=HERE/"control_groups"
folder.mkdir(exist_ok=True)
files=json.loads((V2/"input_manifest.json").read_text())["cells"][cell]["files"]
def read(key):
    file=files[key]
    assert branch.sha(file["path"])==file["sha256"]
    return torch.load(file["path"],map_location="cpu",weights_only=False,mmap=True)
groups,rows=construct(read("checkpoint")["model"],read("scores")["scores"],7,"A",control=True)
branch.save_pt(folder/"sets.pt",groups)
for row in rows:
    row.pop("score_gap")
branch.save_json(folder/"audit.json",dict(objective="seeded uniform first objective per edge",sha256=branch.sha(folder/"sets.pt"),matrices=rows))
