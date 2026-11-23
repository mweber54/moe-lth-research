"""Fail closed if any selected numerical fixture invokes backward/grad/optimizer step."""
_saved=[]
def forbidden(*args,**kwargs):raise AssertionError('Forward-only numerical suite forbids backward/autograd.grad/optimizer steps')
def pytest_configure(config):
    import torch
    for owner,name in [(torch.Tensor,'backward'),(torch.autograd,'backward'),(torch.autograd,'grad')]:
        _saved.append((owner,name,getattr(owner,name)));setattr(owner,name,forbidden)
    for name in dir(torch.optim):
        owner=getattr(torch.optim,name)
        if isinstance(owner,type) and issubclass(owner,torch.optim.Optimizer) and 'step' in owner.__dict__:
            _saved.append((owner,'step',getattr(owner,'step')));setattr(owner,'step',forbidden)
def pytest_unconfigure(config):
    for owner,name,value in reversed(_saved):setattr(owner,name,value)
    _saved.clear()
