"""Train the first NEO skeleton-motion-to-2D-control baseline."""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from neo_video_gen.translator.model import MotionTranslator

class MotionSamples(Dataset):
    def __init__(self,path:Path)->None:
        if not path.exists(): raise FileNotFoundError(path)
        with np.load(path) as data:
            self.joints=data["joints_3d"].astype("float32")
            self.camera=data["camera"].astype("float32")
            self.actions=data["action_id"].astype("int64")
            self.flow=data["target_flow"].astype("float32")
            self.visibility=data["target_visibility"].astype("float32")
            self.depth=data["target_depth_delta"].astype("float32")
        count=len(self.joints)
        if count==0: raise ValueError(f"Dataset split is empty: {path}")
        if not all(len(v)==count for v in (self.camera,self.actions,self.flow,self.visibility,self.depth)):
            raise ValueError("Dataset arrays have inconsistent sample counts")
    def __len__(self): return len(self.joints)
    def __getitem__(self,index):
        return (torch.from_numpy(self.joints[index]),torch.from_numpy(self.camera[index]),
                torch.tensor(self.actions[index],dtype=torch.long),torch.from_numpy(self.flow[index]),
                torch.from_numpy(self.visibility[index]),torch.from_numpy(self.depth[index]))

def evaluate(model,loader,device):
    model.eval(); totals={"flow_l1":0.0,"visibility_bce":0.0,"depth_l1":0.0}; batches=0
    with torch.no_grad():
        for joints,camera,action,flow,visibility,depth in loader:
            prediction=model(joints.to(device),camera.to(device),action.to(device))
            totals["flow_l1"]+=torch.nn.functional.l1_loss(prediction["flow"],flow.to(device)).item()
            totals["visibility_bce"]+=torch.nn.functional.binary_cross_entropy_with_logits(
                prediction["visibility_logits"],visibility.to(device)).item()
            totals["depth_l1"]+=torch.nn.functional.l1_loss(
                prediction["depth_delta"],depth.to(device)).item()
            batches+=1
    return {k:v/max(1,batches) for k,v in totals.items()}

def atomic_save(payload,destination:Path):
    destination.parent.mkdir(parents=True,exist_ok=True)
    temporary=destination.with_suffix(destination.suffix+".tmp")
    torch.save(payload,temporary); temporary.replace(destination)

def make_payload(model,optimizer,history,epoch,best_score,best_epoch,device):
    return {"model_state_dict":model.state_dict(),"optimizer_state_dict":optimizer.state_dict(),
        "config":{"joint_count":17,"camera_dim":32,"action_count":6,"output_size":64},
        "history":history,"epoch":epoch,"best_epoch":best_epoch,
        "best_validation_score":best_score,"device":str(device),
        "note":"Skeleton-motion control baseline; not a dense optical-flow model or FLUX adapter."}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data",type=Path,required=True)
    p.add_argument("--output",type=Path,default=Path("runs/neo_motion_translator.pt"))
    p.add_argument("--epochs",type=int,default=10)
    p.add_argument("--batch-size",type=int,default=16)
    p.add_argument("--lr",type=float,default=2e-4)
    p.add_argument("--workers",type=int,default=0)
    p.add_argument("--device",choices=["auto","cpu","cuda"],default="auto")
    p.add_argument("--patience",type=int,default=12)
    p.add_argument("--min-delta",type=float,default=1e-4)
    p.add_argument("--seed",type=int,default=7)
    p.add_argument("--resume",type=Path,default=None)
    a=p.parse_args()
    if a.epochs<1: raise ValueError("--epochs must be >= 1")
    if a.patience<0 or a.min_delta<0: raise ValueError("--patience and --min-delta must be non-negative")
    if a.device=="cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but no CUDA-enabled PyTorch device is available")
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    device_name="cuda" if a.device=="auto" and torch.cuda.is_available() else a.device
    if device_name=="auto": device_name="cpu"
    device=torch.device(device_name)
    if device.type=="cuda":
        torch.backends.cuda.matmul.allow_tf32=True
        torch.backends.cudnn.benchmark=True
    train=MotionSamples(a.data/"train.npz"); val=MotionSamples(a.data/"val.npz")
    train_loader=DataLoader(train,batch_size=a.batch_size,shuffle=True,num_workers=a.workers,
                             pin_memory=device.type=="cuda")
    val_loader=DataLoader(val,batch_size=a.batch_size,shuffle=False,num_workers=a.workers,
                           pin_memory=device.type=="cuda")
    model=MotionTranslator().to(device); optimizer=torch.optim.AdamW(model.parameters(),lr=a.lr)
    history=[]; best_score=float("inf"); best_epoch=0; stale=0; start=0
    if a.resume:
        if not a.resume.is_file(): raise FileNotFoundError(a.resume)
        saved=torch.load(a.resume,map_location=device,weights_only=False)
        model.load_state_dict(saved["model_state_dict"])
        if "optimizer_state_dict" in saved: optimizer.load_state_dict(saved["optimizer_state_dict"])
        history=list(saved.get("history",[])); start=int(saved.get("epoch",0))
        best_score=float(saved.get("best_validation_score",float("inf")))
        best_epoch=int(saved.get("best_epoch",start))
        if start>=a.epochs: raise ValueError("Resume checkpoint already reaches requested epoch count")
    started=time.monotonic()
    latest=a.output.with_name(a.output.stem+".latest"+a.output.suffix)
    stopped=False
    for epoch in range(start,a.epochs):
        model.train(); total=0.0
        for joints,camera,action,flow,visibility,depth in train_loader:
            joints,camera,action=joints.to(device),camera.to(device),action.to(device)
            flow,visibility,depth=flow.to(device),visibility.to(device),depth.to(device)
            pred=model(joints,camera,action)
            loss=(torch.nn.functional.smooth_l1_loss(pred["flow"],flow)
                +torch.nn.functional.binary_cross_entropy_with_logits(pred["visibility_logits"],visibility)
                +torch.nn.functional.smooth_l1_loss(pred["depth_delta"],depth))
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step()
            total+=float(loss.detach())
        metrics=evaluate(model,val_loader,device)
        score=metrics["flow_l1"]+metrics["visibility_bce"]+metrics["depth_l1"]
        row={"epoch":epoch+1,"train_loss":total/max(1,len(train_loader)),**metrics,"validation_score":score}
        history.append(row); improved=score<best_score-a.min_delta
        if improved: best_score=score; best_epoch=epoch+1; stale=0
        else: stale+=1
        state=make_payload(model,optimizer,history,epoch+1,best_score,best_epoch,device)
        atomic_save(state,latest)
        if improved: atomic_save(state,a.output)
        print(json.dumps(row),flush=True)
        if a.patience and stale>=a.patience:
            stopped=True
            print(f"Early stopping after {stale} epochs without validation improvement.",flush=True)
            break
    report={"device":str(device),"epochs_requested":a.epochs,"epochs_completed":len(history),
        "best_epoch":best_epoch,"best_validation_score":best_score,"early_stopped":stopped,
        "train_samples":len(train),"validation_samples":len(val),
        "elapsed_seconds":time.monotonic()-started,"history":history,
        "model_scope":"skeleton-to-2D control fields; not dense RGB motion or a FLUX adapter"}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.with_suffix(".metrics.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    if not a.output.exists(): raise RuntimeError("No best checkpoint was written")
    print(f"Best checkpoint: {a.output}",flush=True)
    print(f"Latest checkpoint: {latest}",flush=True)
    print(f"Metrics: {a.output.with_suffix('.metrics.json')}",flush=True)

if __name__=="__main__": main()
