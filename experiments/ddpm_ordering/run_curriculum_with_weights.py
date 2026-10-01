#!/usr/bin/env python3
"""Fixed full-data curriculum audit for the locally verified DDPM score norm."""
from __future__ import annotations
import argparse, hashlib, json, os, random, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Sampler

ROOT=Path(__file__).resolve().parent
PROXY=Path(os.environ.get('COHE_DDPM_PROXY', str(Path(__file__).resolve().parents[2] / 'derived_scalar_arrays/cifar100/arrays/ddpm_scores.npy')))
DATA=Path(os.environ.get('COHE_CIFAR_ROOT', 'user_inputs/cifar100'))
EVAL_EPOCHS=tuple(range(20,201,20))

def sha256(path:Path)->str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()

def set_seed(seed:int)->None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=False; torch.backends.cudnn.benchmark=True

def model()->nn.Module:
    return torchvision.models.resnet18(weights=None,num_classes=100)

def build_data():
    tr=transforms.Compose([transforms.RandomCrop(32,padding=4),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize((0.5071,0.4867,0.4408),(0.2675,0.2565,0.2761))])
    te=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5071,0.4867,0.4408),(0.2675,0.2565,0.2761))])
    train=torchvision.datasets.CIFAR100(root=str(DATA),train=True,download=False,transform=tr)
    test=torchvision.datasets.CIFAR100(root=str(DATA),train=False,download=False,transform=te)
    return train,test

class FixedOrderSampler(Sampler[int]):
    def __init__(self,n:int): self.order=np.arange(n,dtype=np.int64)
    def set_order(self,order:np.ndarray)->None: self.order=np.asarray(order,dtype=np.int64)
    def __iter__(self): return iter(self.order.tolist())
    def __len__(self): return int(len(self.order))

def load_bins(scores_override=None):
    scores=np.asarray(np.load(PROXY) if scores_override is None else scores_override,dtype=np.float64)
    if scores.shape!=(50000,) or not np.isfinite(scores).all(): raise ValueError('proxy shape/finite check failed')
    ids=np.arange(len(scores),dtype=np.int64)
    order=np.lexsort((ids,scores))
    bins=[np.asarray(x,dtype=np.int64) for x in np.array_split(order,10)]
    if any(len(x)!=5000 for x in bins): raise ValueError('non-equal bins')
    return scores,bins

def make_order(policy:str,seed:int,epoch:int,bins:list[np.ndarray],n:int)->np.ndarray:
    rng=np.random.RandomState(seed*1000003+epoch*10007+(0 if policy=='EASY_TO_HARD' else 50000000))
    if policy=='RANDOM': return rng.permutation(n).astype(np.int64)
    parts=[rng.permutation(b) for b in bins]
    if policy=='HARD_TO_EASY': parts=parts[::-1]
    return np.concatenate(parts).astype(np.int64)

def accuracy(net,loader,device):
    net.eval(); good=total=0
    with torch.no_grad():
        for x,y in loader:
            x=x.to(device,non_blocking=True); y=y.to(device,non_blocking=True)
            good+=int((net(x).argmax(1)==y).sum().item()); total+=int(y.numel())
    return 100.0*good/total

def run(policy:str,seed:int,out_dir:Path,device_name:str):
    base_scores=np.asarray(np.load(PROXY),dtype=np.float64)
    permutation=None
    if policy=='SHUFFLED_PROXY_ORDER':
        permutation=np.random.RandomState(seed).permutation(len(base_scores)).astype(np.int64)
        scores=base_scores[permutation]
    else:
        scores=base_scores
    scores,bins=load_bins(scores); n=len(scores); set_seed(seed)
    out_dir.mkdir(parents=True,exist_ok=True)
    train,test=build_data(); sampler=FixedOrderSampler(n)
    loader=DataLoader(train,batch_size=128,sampler=sampler,shuffle=False,num_workers=4,pin_memory=True,persistent_workers=True)
    test_loader=DataLoader(test,batch_size=100,shuffle=False,num_workers=4,pin_memory=True,persistent_workers=True)
    device=torch.device(device_name if torch.cuda.is_available() and device_name.startswith('cuda') else 'cpu')
    net=model().to(device); opt=optim.SGD(net.parameters(),lr=.1,momentum=.9,weight_decay=5e-4); sch=optim.lr_scheduler.CosineAnnealingLR(opt,T_max=200); loss_fn=nn.CrossEntropyLoss()
    first=make_order(policy,seed,1,bins,n); np.save(out_dir/'first_epoch_order.npy',first)
    if len(np.unique(first))!=n or set(first.tolist())!=set(range(n)): raise ValueError('order verification failed')
    if permutation is not None:
        np.save(out_dir/'permutation.npy',permutation)
        np.save(out_dir/'shuffled_score_assignment.npy',scores)
    verification={'policy':policy,'seed':seed,'n':n,'proxy':str(PROXY),'proxy_sha256':sha256(PROXY),'n_bins':10,'bin_size':5000,'first_epoch_order_sha256':sha256(out_dir/'first_epoch_order.npy'),'exactly_once':True,'ordering':'ascending DDPM score norm, easy-to-hard' if policy in ('EASY_TO_HARD','SHUFFLED_PROXY_ORDER') else 'canonical global random order per epoch','permutation_seed':seed if permutation is not None else None,'permutation_sha256':sha256(out_dir/'permutation.npy') if permutation is not None else None,'shuffled_score_assignment_sha256':sha256(out_dir/'shuffled_score_assignment.npy') if permutation is not None else None}
    (out_dir/'order_verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    losses=[]; evals={}; t0=time.time()
    for epoch in range(1,201):
        sampler.set_order(make_order(policy,seed,epoch,bins,n)); net.train(); running=0.0
        for x,y in loader:
            x=x.to(device,non_blocking=True); y=y.to(device,non_blocking=True); opt.zero_grad(set_to_none=True); loss=loss_fn(net(x),y); loss.backward(); opt.step(); running+=float(loss.item())
        sch.step(); mean_loss=running/len(loader); losses.append(mean_loss)
        if epoch in EVAL_EPOCHS: evals[f'acc_epoch_{epoch}']=accuracy(net,test_loader,device)
        if epoch%20==0: print(f'{policy} seed={seed}: epoch {epoch:03d}/200 loss={mean_loss:.4f} acc={evals[f"acc_epoch_{epoch}"]:.3f}',flush=True)
    torch.save({'model_state_dict':net.state_dict(),'optimizer_state_dict':opt.state_dict(),'scheduler_state_dict':sch.state_dict(),'epoch':200},out_dir/'model_epoch200.pt')
    payload={'status':'completed','policy':policy,'seed':seed,'proxy':str(PROXY),'proxy_sha256':sha256(PROXY),'dataset':'CIFAR-100','n':n,'bins':10,'bin_size':5000,'score_orientation':'higher DDPM score norm = harder; bins ascending are easy-to-hard','batch_size':128,'epochs':200,'lr':.1,'momentum':.9,'weight_decay':5e-4,'optimizer':'SGD','scheduler':'CosineAnnealingLR(T_max=200)','eval_epochs':list(EVAL_EPOCHS),'training_loss':losses,'test_accuracy':evals,'acc_final':float(evals['acc_epoch_200']),'runtime_sec':time.time()-t0,'device':str(device),'order_verification':str(out_dir/'order_verification.json'),'model_checkpoint':str(out_dir/'model_epoch200.pt'),'optimizer_steps':200*len(loader),'num_workers':4,'augmentation':'RandomCrop(32,padding=4)+RandomHorizontalFlip','normalization':'CIFAR-100 mean/std','split':'train=50000,test=10000'}
    (out_dir/'result.json').write_text(json.dumps(payload,indent=2)+'\n')
    print(json.dumps({'status':'completed','policy':policy,'seed':seed,'acc_final':payload['acc_final'],'runtime_sec':payload['runtime_sec']}),flush=True)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--policy',choices=['EASY_TO_HARD','HARD_TO_EASY','RANDOM','SHUFFLED_PROXY_ORDER'],required=True); ap.add_argument('--seed',type=int,required=True); ap.add_argument('--out-dir',type=Path,required=True); ap.add_argument('--device',default='cuda'); a=ap.parse_args(); run(a.policy,a.seed,a.out_dir,a.device)
if __name__=='__main__': main()
