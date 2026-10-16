import torch
from model.channel_kd import channel_foreground_kd,relation_selection
from model.selective_kd import old_foreground_kd
torch.set_num_threads(1)
s=torch.tensor([[[[0.,1.]],[[0.,3.]],[[2.,0.]],[[9.,9.]]]],dtype=torch.float64,requires_grad=True)
t=torch.tensor([[[[0.,0.]],[[4.,4.]],[[-4.,-4.]],[[0.,0.]]]],dtype=torch.float64,requires_grad=True)
e=torch.tensor([[[True,False]]]);old=torch.tensor([[[1,1]]]);zero=torch.zeros_like(e)
m={'eligible':e,'selected':e,'new_supported':zero}
original,_=old_foreground_kd(s,t,e,e,2,2.)
x,_,stats=channel_foreground_kd(s,t,old,m,2,'counter')
assert torch.allclose(x,original,atol=1e-12,rtol=1e-12)
g0=torch.autograd.grad(original,s,retain_graph=True)[0];g1=torch.autograd.grad(x,s,retain_graph=True)[0];assert torch.allclose(g0,g1,atol=1e-12)
x.backward();assert t.grad is None
assert s.grad[:,0].abs().sum()==0 and s.grad[:,3].abs().sum()==0 and s.grad[:,:,:,1].abs().sum()==0
s.grad=None;m={'eligible':e,'selected':zero,'new_supported':e}
x,_,stats=channel_foreground_kd(s,t,old,m,2,'counter');x.backward(retain_graph=True)
assert s.grad[:,1].abs().sum()==0 and s.grad[:,2,:,:,].abs().sum()>0
assert stats['kd_rejected_positive_channel_sum']==1 and stats['kd_selected_channel_sum']==1
counter_grad=s.grad[:,2].clone();s.grad=None
x,_,stats=channel_foreground_kd(s,t,old,m,2,'relations');x.backward(retain_graph=True)
assert s.grad[:,1].abs().sum()==0 and torch.allclose(s.grad[:,2],2*counter_grad)
assert stats['kd_channel_weight_sum']==stats['kd_eligible_channel_sum']==2
expected=(s.detach()[:,2]/2).sigmoid()-(t.detach()[:,2]/2).sigmoid()
assert torch.allclose(s.grad[:,2,:,:1],(4*expected)[:,:,:1],atol=1e-12)
s.grad=None
ambiguous=torch.zeros_like(t,requires_grad=True)
x,_,stats=channel_foreground_kd(s,ambiguous,old,m,2,'relations');x.backward()
assert x==0 and torch.isfinite(x) and s.grad.abs().sum()==0 and ambiguous.grad is None
m={'eligible':e,'selected':e,'new_supported':zero};both=t.detach().clone();both[:,2]=4.
sel,p=relation_selection(both,old,m,2,'relations');assert sel[:,1].sum()==0 and sel[:,0].sum()==1
print('PASS: counter without contradiction exactly matches ordinaryKD value/gradient; teacher/BG/new/ignored gradients excluded; veto removes positive but preserves negative gradient; relations matches channel-weight budget and analytic negative gradient; empty selection finitezero; only licensed old winning positive retained.')
