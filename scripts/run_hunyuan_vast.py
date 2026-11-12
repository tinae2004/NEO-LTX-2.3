#!/usr/bin/env python3
"""Budget-capped Vast runner for HunyuanVideo 1.5 T2V."""
from __future__ import annotations
import argparse, os, re, shlex, shutil, subprocess, sys, tempfile, time
from pathlib import Path
import requests

API="https://console.vast.ai/api/v0"
def run(cmd, timeout=60, check=True, **kw):
    return subprocess.run(cmd, text=True, timeout=timeout, check=check, **kw)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--prompt",required=True)
    p.add_argument("--budget-usd",type=float,default=0.50)
    p.add_argument("--max-hourly-usd",type=float,default=0.10)
    p.add_argument("--max-hours",type=float,default=1.0)
    p.add_argument("--disk-gb",type=int,default=160)
    p.add_argument("--output-dir",default="hunyuan-output")
    a=p.parse_args()
    if not (0<a.max_hourly_usd<=0.10 and 0<a.budget_usd<=1.00 and 0.25<=a.max_hours<=1.0):
        raise SystemExit("Safety cap: hourly <= $0.10, total budget <= $1.00, runtime 0.25-1.0 hour.")
    if a.max_hourly_usd*a.max_hours*1.25>a.budget_usd:
        raise SystemExit("Budget does not cover hourly cap x runtime plus 25% reserve.")
    key=os.environ.get("VAST_API_KEY","").strip()
    if not key: raise SystemExit("VAST_API_KEY secret is missing.")
    s=requests.Session(); s.headers.update({"Authorization":f"Bearer {key}","Content-Type":"application/json"})
    def api(method,path,payload=None):
        r=s.request(method,API+path,json=payload,timeout=45)
        if not r.ok: raise RuntimeError(f"Vast API {method} {path}: HTTP {r.status_code} {r.text[:500]}")
        return r.json() if r.text.strip() else {}
    offers=api("POST","/bundles/",{
        "gpu_name":{"in":["RTX 5060 Ti","RTX 5070 Ti","RTX 4070 Ti SUPER","RTX 4080","RTX 5080","RTX 4060 Ti"]},
        "num_gpus":{"eq":1},"gpu_ram":{"gte":14},"cpu_ram":{"gte":48},
        "disk_space":{"gte":a.disk_gb},"reliability":{"gte":0.99},
        "verified":{"eq":True},"rentable":{"eq":True},"type":"ondemand","limit":50
    }).get("offers",[])
    candidates=[]
    for o in offers:
        name=str(o.get("gpu_name","")).upper().replace("_"," ")
        rate=float(o.get("dph_total",o.get("dph",0)) or 0)
        if not any(n in name for n in ("RTX 5060 TI","RTX 5070 TI","RTX 4070 TI SUPER","RTX 4080","RTX 5080","RTX 4060 TI")): continue
        if rate<=0 or rate>a.max_hourly_usd or rate*a.max_hours*1.25>a.budget_usd: continue
        candidates.append((rate,-float(o.get("reliability",0) or 0),o))
    candidates.sort(key=lambda x:(x[0],x[1]))
    if not candidates: raise SystemExit("No verified 16GB RTX offer meets all limits. No rental made.")
    out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    temp=Path(tempfile.mkdtemp(prefix="neo-hunyuan-")); instance=None; agent=False
    try:
        env=os.environ.copy()
        ag=run(["ssh-agent","-s"],capture_output=True)
        for k,v in re.findall(r"(SSH_AUTH_SOCK|SSH_AGENT_PID)=([^;]+);",ag.stdout): env[k]=v
        if "SSH_AUTH_SOCK" not in env: raise RuntimeError("ssh-agent failed")
        agent=True
        run(["vastai","set","api-key",key],env=env)\n        kp=temp/"key"; run(["ssh-keygen","-q","-t","ed25519","-N","","-f",str(kp)])
        run(["vastai","create","ssh-key",Path(str(kp)+".pub").read_text().strip()],env=env)
        selected=None; last=None
        for _,__,offer in candidates[:12]:
            try:
                resp=api("PUT",f"/asks/{int(offer['id'])}/",{
                    "image":"vastai/pytorch:@vastai-automatic-tag","label":"NEO-Hunyuan-T2V-16GB",
                    "disk":a.disk_gb,"runtype":"ssh_direct","cancel_unavail":True})
                instance=resp.get("new_contract")
                if instance is None: raise RuntimeError(f"Rental response missing instance id: {resp}")
                selected=offer; break
            except Exception as e:
                last=e; print(f"Offer {offer.get('id')} failed: {e}",flush=True)
        if instance is None: raise RuntimeError(f"No offer could be rented. Last error: {last}")
        rate=float(selected.get("dph_total",selected.get("dph")))
        print(f"RENTED instance={instance} gpu={selected.get('gpu_name')} hourly_usd={rate:.4f}",flush=True)
        deadline=time.monotonic()+a.max_hours*3600
        def get_instance():
            d=api("GET",f"/instances/{instance}/"); x=d.get("instances",d)
            if isinstance(x,list): return next((z for z in x if int(z.get("id",-1))==int(instance)),{})
            return x if isinstance(x,dict) else {}
        ready_deadline=min(deadline,time.monotonic()+1200)
        while time.monotonic()<ready_deadline:
            st=get_instance(); status=str(st.get("actual_status",st.get("status",""))).lower()
            print(f"INSTANCE_STATUS={status}",flush=True)
            if status=="running": break
            if status in ("dead","error","exited","deleted","stopped"): raise RuntimeError(f"Instance failed: {status}")
            time.sleep(15)
        else: raise TimeoutError("GPU did not become running within 20 minutes.")
        ssh_base=None; ssh_deadline=min(deadline,time.monotonic()+600)
        while time.monotonic()<ssh_deadline:
            value=run(["vastai","ssh-url",str(instance)],env=env,capture_output=True).stdout.strip()
            opts=["-i",str(kp),"-o","IdentitiesOnly=yes","-o","StrictHostKeyChecking=no","-o","UserKnownHostsFile=/dev/null","-o","ConnectTimeout=12"]
            if value.startswith("ssh://"):
                from urllib.parse import urlparse
                u=urlparse(value); ssh_base=["ssh",*opts,"-p",str(u.port),f"{u.username or 'root'}@{u.hostname}"]
            else:
                toks=shlex.split(value); toks=toks[1:] if toks and toks[0]=="ssh" else toks
                ssh_base=["ssh",*opts,*toks] if toks else None
            if ssh_base:
                probe=run(ssh_base+["nvidia-smi"],env=env,capture_output=True,check=False,timeout=30)
                if probe.returncode==0: print(probe.stdout,flush=True); break
                ssh_base=None
            time.sleep(10)
        if not ssh_base: raise TimeoutError("SSH/NVIDIA preflight never became ready.")
        source=subprocess.Popen(["git","archive","--format=tar.gz","HEAD"],stdout=subprocess.PIPE)
        recv=subprocess.Popen(ssh_base+["bash -lc "+shlex.quote("mkdir -p /workspace/neo && tar -xzf - -C /workspace/neo")],stdin=source.stdout)
        source.stdout.close()
        if recv.wait(timeout=300)!=0 or source.wait(timeout=60)!=0: raise RuntimeError("Source transfer failed.")
        remote_cmd="cd /workspace/neo && chmod +x scripts/run_hunyuan_t2v_16gb.sh && ./scripts/run_hunyuan_t2v_16gb.sh "+shlex.quote(a.prompt)
        remaining=max(60,int(deadline-time.monotonic()))
        print("STARTING_HUNYUAN_GENERATION",flush=True)
        result=run(ssh_base+["bash -lc "+shlex.quote(remote_cmd)],env=env,timeout=remaining,capture_output=True,check=False)
        print(result.stdout[-12000:],flush=True)
        if result.stderr: print(result.stderr[-6000:],file=sys.stderr,flush=True)
        if result.returncode: raise RuntimeError(f"Hunyuan generation exited with code {result.returncode}")
        found=run(ssh_base+["bash -lc "+shlex.quote("find /workspace/neo-data/outputs/hunyuan -maxdepth 1 -type f -name '*.mp4' -print -quit")],env=env,capture_output=True).stdout.strip()
        if not found: raise RuntimeError("Generation completed but no MP4 was found.")
        local=out/"hunyuan_t2v.mp4"
        with local.open("wb") as f:
            cp=run(ssh_base+["cat "+shlex.quote(found)],env=env,stdout=f,stderr=subprocess.PIPE,timeout=300,check=False)
        if cp.returncode or local.stat().st_size==0: raise RuntimeError("Failed to download generated MP4.")
        print(f"VIDEO_OUTPUT={local} BYTES={local.stat().st_size}",flush=True)
    finally:
        if instance is not None:
            try:
                print(f"DESTROYING_INSTANCE={instance}",flush=True); api("DELETE",f"/instances/{instance}/")
                print("DESTROY_REQUEST_ACCEPTED",flush=True)
            except Exception as e: print(f"CRITICAL_DESTROY_FAILURE instance={instance}: {e}",file=sys.stderr,flush=True)
        if agent: run(["ssh-agent","-k"],env=env,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,check=False)
        shutil.rmtree(temp,ignore_errors=True)

if __name__=="__main__": main()
