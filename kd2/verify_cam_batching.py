import torch,json
from model.model_seg_neg import network
from model.cam_batching import memory_bounded_cams
from utils.camutils import multi_scale_cam2
torch.set_num_threads(2);torch.manual_seed(17)
net=network('deit_base_patch16_224',num_classes=16,classes_list=[11,5],pretrained=False,aux_layer=-3,step=1)
net.train()
assert not any(isinstance(m,torch.nn.modules.batchnorm._BatchNorm) for m in net.modules())
assert all(m.p==0 for m in net.modules() if isinstance(m,torch.nn.Dropout))
assert all(getattr(m,'drop_prob',0)==0 for m in net.modules())
images=torch.randn(8,3,64,64)
rng=torch.random.get_rng_state().clone()
with torch.no_grad(): direct=multi_scale_cam2(net,images,[1.,.5,1.5]);chunked=memory_bounded_cams(net,images,[1.,.5,1.5],4)
assert torch.equal(rng,torch.random.get_rng_state())
errors=[]
for a,b in zip(direct,chunked):
 assert a.shape==b.shape and not b.requires_grad
 assert torch.allclose(a,b,atol=1e-5,rtol=1e-4),(a-b).abs().max().item()
 errors.append((a-b).abs().max().item())
assert all(p.grad is None for p in net.parameters())
print(json.dumps({'status':'pass','CAM_main_aux_max_abs_difference':errors,'model':'actualDeiTbase12blocks_trainmode','logical_images':8,'CAM_chunk':4,'scales':[1,.5,1.5],'RNG_unchanged':True,'no_batchnorm_or_active_dropout':True,'no_grad':True}))
