"""Meaningful checks for KD masks, gradients, teacher detachment and normalization."""
import torch
from torch.nn import functional as F
from model.selective_kd import kd_evidence, old_foreground_kd

torch.manual_seed(0)
s=torch.randn(1,4,2,3,requires_grad=True)
t=torch.randn(1,3,2,3,requires_grad=True)
eligible=torch.tensor([[[True,True,True],[True,True,False]]])
selected=torch.tensor([[[True,False,True],[False,True,False]]])
loss,pixel=old_foreground_kd(s,t,selected,eligible,2,2.)
p=(t.detach()[:,1:3]/2).sigmoid()
reference=(F.binary_cross_entropy_with_logits(s[:,1:3]/2,p,reduction='none')
           -F.binary_cross_entropy_with_logits(t.detach()[:,1:3]/2,p,reduction='none')).sum(1)*4
torch.testing.assert_close(loss,(reference*selected).sum()/eligible.sum())
loss.backward()
assert t.grad is None
assert torch.count_nonzero(s.grad[:,0])==0 and torch.count_nonzero(s.grad[:,3])==0
assert torch.count_nonzero(s.grad[:,1:3]*~selected[:,None])==0
assert s.grad[:,1:3].abs().sum()>0
z=torch.randn(1,4,2,3,requires_grad=True)
empty,_=old_foreground_kd(z,t,torch.zeros_like(selected),eligible,2)
empty.backward()
assert empty.item()==0 and torch.count_nonzero(z.grad)==0
same,_=old_foreground_kd(torch.cat([t.detach(),torch.zeros(1,1,2,3)],1),t,eligible,eligible,2)
assert abs(same.item())<1e-7

old=torch.tensor([[[1,1,1,1,0,1,1,1]]])
target=torch.tensor([[[1,3,3,3,0,1,255,1]]])
valid=torch.tensor([[[True,True,True,True,True,True,True,False]]])
teacher=torch.full((1,3,1,8),-4.); teacher[:,1]=4.;teacher[:,1,:,5]=0
def mask(indices):
 a=torch.zeros_like(valid)
 for i in indices:a[0,0,i]=True
 return a
paired={'conflict':mask([1,2,3]),'to_old':mask([1]),'new_strong':mask([2]),
        'old_score':torch.ones_like(target,dtype=torch.float)}
e=kd_evidence(teacher,old,target,valid,paired,mask([4]),2,.75)
assert torch.equal(e['selected'],mask([0,1]))
assert torch.equal(e['new_supported'],mask([2,4]))
assert torch.equal(e['ambiguous'],mask([3]))
assert torch.equal(e['unresolved'],mask([3,5]))
assert not (e['selected'] & e['new_supported']).any()
assert torch.equal(e['selected'] | e['new_supported'] | e['unresolved'],e['eligible'])
assert not (e['selected'] & mask([4,5,6,7])).any()
print('PASS: Bernoulli KL reference; teacher detached; background/new-channel/ignored gradients zero; empty gate; evidence partition; uncertain teacher exclusion.')
