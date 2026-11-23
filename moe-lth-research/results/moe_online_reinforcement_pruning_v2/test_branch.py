"""Focused CPU regressions for selection mapping, permanent masks and state."""
import torch
import branch
assert branch.ARMS=={'untouched':None,'random-small':'random','high-reinforcement-small':'high'}
model=torch.nn.Sequential(torch.nn.Linear(512,512,bias=False));opt=torch.optim.AdamW(model.parameters(),lr=.001)
p=next(model.parameters());(p.square().sum()).backward();opt.step();opt.zero_grad(set_to_none=True)
name='0.weight';idx=torch.arange(13107);selection={name:idx};before=p.detach().clone();moment=opt.state[p]['exp_avg'].clone();step=opt.state[p]['step'].clone();mask=branch.build_masks(model,opt,selection)
branch.enforce(model,opt,mask);assert torch.equal(p.view(-1)[13107:],before.view(-1)[13107:]);assert not torch.count_nonzero(p.view(-1)[:13107]);assert torch.equal(opt.state[p]['exp_avg'].view(-1)[13107:],moment.view(-1)[13107:]);assert torch.equal(opt.state[p]['step'],step)
p.grad=torch.ones_like(p);branch.enforce(model,opt,mask,grad=True);assert not torch.count_nonzero(p.grad.view(-1)[:13107]);opt.step();branch.enforce(model,opt,mask);branch.verify_zeros(model,opt,mask)
try:branch.build_masks(model,opt,{name:torch.zeros(13107,dtype=torch.long)})
except RuntimeError:pass
else:raise AssertionError('duplicate selections accepted')
print('mask and optimizer regression checks passed')
