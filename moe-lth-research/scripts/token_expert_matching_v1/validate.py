"""Executable canonical CUDA validation; all output confined to this experiment."""
from run import *
import copy


def equal(a,b):
    if torch.is_tensor(a):return torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b,strict=True))
    return a==b


def main():
    torch.set_num_threads(4);device=torch.device('cuda');R.configure_device(device)
    host=socket.getfqdn()
    if host not in HOSTS or not torch.__version__.startswith('2.7.0'):raise RuntimeError('Canonical runtime required')
    if torch.cuda.mem_get_info()[0]<24*1024**3:raise RuntimeError('GPU admission failed')
    base=DEST/'validation'/host/f'attempt_{time.time_ns()}';base.mkdir(parents=True)
    manifest=frozen();v=next(x for x in manifest['variants'] if x['seed']==17 and x['sparsity']==.8)
    train,val=batches(v);masks=load_masks(v['mask']);m=load_model_from_checkpoint(v['config']['model'],v['initial'],torch.device('cpu'));apply_masks_(m,masks);initial=model_state(m);del m
    schedules={};checks=[]
    def traced(model,tokens,step):
        before=torch.cuda.get_rng_state_all()
        output=R._condition_forward(model,tokens,False,1.)
        after=torch.cuda.get_rng_state_all();cpu=torch.get_rng_state().clone()
        payload=replay.capture(model,output,tokens,step,R._batch_sequence_hash([train[step-1]]))
        assert equal(after,torch.cuda.get_rng_state_all()) and torch.equal(cpu,torch.get_rng_state())
        schedules[step]=payload
        # Clone under protection; replay consumes exactly the same dropout RNG.
        with protected(model):
            clone=TinyMoELanguageModel(v['config']['model']).to(device);clone.load_state_dict(model.state_dict());replay.install(clone)
            torch.cuda.set_rng_state_all(before)
            alternate=replay.forward(clone,tokens,payload,step,payload['batch_hash'])
            assert torch.equal(output.logits,alternate.logits),'Identity forward mismatch'
            assert torch.equal(output.auxiliary_loss,alternate.auxiliary_loss)
            for x,y in zip(output.routes,alternate.routes,strict=True):assert torch.equal(x.accepted_mask,y.accepted_mask)
            for n,p in clone.named_parameters():p.requires_grad_('.router.' not in n)
            replayed=replay.forward(clone,tokens,payload,step,payload['batch_hash'],101,17)
            assert any(not torch.equal(x.selected_expert_indices,y.selected_expert_indices) for x,y in zip(output.routes,replayed.routes,strict=True))
            loss=torch.nn.functional.cross_entropy(replayed.logits.flatten(0,1),train[step-1][1].to(device).flatten())
            loss.backward(retain_graph=True)
            assert all(p.grad is None for n,p in clone.named_parameters() if '.router.' in n)
            assert any(p.grad is not None and p.grad.norm()>0 for n,p in clone.named_parameters() if '.experts.' in n)
            assert clone.token_embedding.weight.grad.norm()>0
            clone.zero_grad();replayed.auxiliary_loss.backward()
            assert clone.token_embedding.weight.grad.norm()>0
            del clone,alternate,replayed
        checks.append(dict(step=step,exact_identity_forward=True,exact_acceptance=True,gradients=True,rng_preserved=True))
        return output
    records=[]
    for label,fn in [('plain',None),('trace',traced)]:
        records.append(segment(v['config'],initial,v['initial'],masks,train[:3],val[:1],base/label,'full',17,forward=fn,device=device,resume_interval=1))
    a=load(base/'plain/interruption.pt');b=load(base/'trace/interruption.pt')
    for key in ('model','optimizer','scaler','python_rng','numpy_rng','torch_rng','cuda_rng','step'):assert equal(a[key],b[key]),key
    # Replay interrupted at step 3, restore completed step 2 and deterministic permutation.
    def replayed(model,tokens,step):
        return replay.forward(model,tokens,schedules[step],step,schedules[step]['batch_hash'],101,17)
    def interrupt(model,opt,scaler,step):
        if step==3:raise RuntimeError('simulated interruption')
    kw=dict(config=v['config'],initial=initial,router_path=v['initial'],masks=masks,train=train[:3],val=val[:1],condition='expert_shared',seed=17,forward=replayed,assembled=replay.install,device=device,resume_interval=1)
    segment(dest=base/'replay_full',**kw)
    try:segment(dest=base/'replay_resume',callback=interrupt,**kw)
    except RuntimeError as error:
        if str(error)!='simulated interruption':raise
    else:raise AssertionError('Interruption did not occur')
    segment(dest=base/'replay_resume',**kw)
    a=load(base/'replay_full/interruption.pt');b=load(base/'replay_resume/interruption.pt')
    for key in ('model','optimizer','scaler','python_rng','numpy_rng','torch_rng','cuda_rng','step'):assert equal(a[key],b[key]),key
    boundary=load(base/'replay_full/checkpoints/final_recovered.pt')['model'];seen=[]
    segment(v['config'],boundary,base/'replay_full/checkpoints/final_recovered.pt',masks,train[:1],val[:1],base/'reset','expert_shared',17,offset=2500,forward=lambda m,t,s:replay.forward(m,t,schedules[1],1,schedules[1]['batch_hash'],101,17),assembled=replay.install,callback=lambda m,o,s,i:seen.append(dict(hash=state_dict_hash(m.state_dict()),optimizer_entries=len(o.state),scaler=s.state_dict())) if i==0 else None,device=device)
    assert seen[0]['hash']==state_dict_hash(boundary) and seen[0]['optimizer_entries']==0
    result=dict(passed=True,host=host,torch=str(torch.__version__),checks=checks,passive_trace_exact_state_rng_optimizer_scaler=True,replay_resume_exact=True,segment_reset=seen[0],attempt=str(base))
    save(DEST/'validation'/f'{host}.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
