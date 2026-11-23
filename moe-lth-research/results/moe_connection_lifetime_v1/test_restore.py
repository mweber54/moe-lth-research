"""Detect a donor update mutating another donor's starting Adam counters."""
import copy
import torch
from lifetime_engineering import restore,branch


def main():
    model=torch.nn.Linear(3,2)
    opt=torch.optim.AdamW(model.parameters())
    scaler=branch.create_grad_scaler(False)
    def advance():
        opt.zero_grad()
        model(torch.ones(1,3)).sum().backward()
        opt.step()
    advance()
    # A CPU fixture exercises aliases for both steps and moment tensors.
    saved=copy.deepcopy(dict(model=model.state_dict(),optimizer=opt.state_dict(),scaler=scaler.state_dict(),**branch.rng_get()))
    initial=copy.deepcopy(saved["optimizer"])
    restore(saved,model,opt,scaler)
    advance()
    assert branch.nested_equal(saved["optimizer"],initial)
    assert all(int(s["step"])==2 for s in opt.state.values())


if __name__=="__main__":
    main()
