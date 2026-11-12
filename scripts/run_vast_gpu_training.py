"""Provision a budget-capped Vast.ai GPU and train NEO from a GitHub Actions runner."""
from __future__ import annotations
import argparse, json, os, re, shlex, shutil, subprocess, sys, tarfile, tempfile, time
from pathlib import Path
import requests
from urllib.parse import urlparse

API_BASE="https://console.vast.ai/api/v0"
BLENDER_VERSION="4.5.16"

class VastAPI:
    def __init__(self,key):
        self.session=requests.Session()
        self.session.headers.update({"Authorization":f"Bearer {key}","Content-Type":"application/json"})
    def request(self,method,path,payload=None):
        response=self.session.request(method,API_BASE+path,json=payload,timeout=45)
        if not response.ok:
            raise RuntimeError(f"Vast API {method} {path} failed ({response.status_code}): {response.text[:800]}")
        if response.status_code==204 or not response.text.strip(): return {}
        return response.json()
    def search(self,names):
        return self.request("POST","/bundles/",{
            "gpu_name":{"in":names},"num_gpus":{"eq":1},"gpu_ram":{"gte":24},
            "disk_space":{"gte":80},"reliability":{"gte":0.99},"direct_port_count":{"gte":1},
            "verified":{"eq":True},"rentable":{"eq":True},"type":"ondemand","limit":20
        }).get("offers",[])
    def create(self,offer_id,disk_gb,label):
        return self.request("PUT",f"/asks/{offer_id}/",{
            "image":"vastai/pytorch:@vastai-automatic-tag",
            "label":label,"disk":disk_gb,"runtype":"ssh_direct","cancel_unavail":True
        })
    def get_instance(self,instance_id):
        result=self.request("GET",f"/instances/{instance_id}/")
        instance=result.get("instances",result)
        if isinstance(instance,list):
            instance=next((x for x in instance if int(x.get("id",-1))==instance_id),{})
        return instance if isinstance(instance,dict) else {}
    def destroy(self,instance_id):
        return self.request("DELETE",f"/instances/{instance_id}/")

def run_local(command,env=None,timeout=60,capture=True):
    result=subprocess.run(command,env=env,text=True,timeout=timeout,
        stdout=subprocess.PIPE if capture else None,stderr=subprocess.PIPE if capture else None)
    if result.returncode:
        detail=(result.stderr or result.stdout or "")[-2500:] if capture else ""
        raise RuntimeError(f"Command failed ({result.returncode}): {command!r}\n{detail}")
    return (result.stdout or "").strip() if capture else ""

def start_agent(env):
    result=subprocess.run(["ssh-agent","-s"],text=True,stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE,check=True)
    for key,value in re.findall(r"(SSH_AUTH_SOCK|SSH_AGENT_PID)=([^;]+);",result.stdout):
        env[key]=value
    if "SSH_AUTH_SOCK" not in env: raise RuntimeError("Could not start ssh-agent")
    return env

def parse_usd(value):
    """Accept numeric inputs such as 5, $5, or $0.30 from the Actions form."""
    normalized = str(value).strip().replace("$", "").replace(",", "")
    return float(normalized)

def as_float(offer,key,default=0.0):
    try: return float(offer.get(key,default) or default)
    except (TypeError,ValueError): return default

def eligible_offers(offers,gpu_order,max_hourly,budget,max_hours):
    wanted=[x.strip().replace("_"," ").upper() for x in gpu_order.split(",") if x.strip()]
    ranked=[]
    for offer in offers:
        name=str(offer.get("gpu_name","")).replace("_"," ").upper()
        rank=next((i for i,item in enumerate(wanted) if item in name),None)
        rate=as_float(offer,"dph_total",as_float(offer,"dph"))
        reliability=as_float(offer,"reliability")
        if rank is None or rate<=0 or rate>max_hourly or reliability<0.99: continue
        estimate=rate*max_hours*1.25
        if estimate>budget: continue
        ranked.append((rank,rate,-reliability,offer))
    ranked.sort(key=lambda row:(row[0],row[1],row[2]))
    return [row[3] for row in ranked]

def rent_with_refresh(api, gpu_names, gpu_order, max_hourly, budget, max_hours,
                       disk_gb, label, max_searches=5, offers_per_search=8):
    """Rent from fresh offer snapshots, excluding stale IDs and respecting all caps."""
    excluded_ids=set()
    attempts=0
    last_error=None
    for search_no in range(1,max_searches+1):
        offers=api.search(gpu_names)
        candidates=eligible_offers(offers,gpu_order,max_hourly,budget,max_hours)
        candidates=[o for o in candidates if str(o.get("id")) not in excluded_ids]
        print(f"Vast offer refresh {search_no}/{max_searches}: "
              f"{len(offers)} returned, {len(candidates)} eligible after exclusions",flush=True)
        if not candidates:
            if search_no < max_searches:
                time.sleep(min(2*search_no,8))
                continue
            break
        refresh_now=False
        for offer in candidates[:offers_per_search]:
            offer_id=str(offer.get("id"))
            if not offer_id or offer_id == "None":
                continue
            attempts += 1
            rate=as_float(offer,"dph_total",as_float(offer,"dph"))
            print(f"Attempting offer={offer_id} gpu={offer.get('gpu_name')} "
                  f"rate_usd_per_hour={rate:.4f} reliability={offer.get('reliability')}",flush=True)
            try:
                response=api.create(int(offer_id),disk_gb,label)
                contract=response.get("new_contract")
                if contract is None:
                    raise RuntimeError(f"No instance ID in Vast response: {response}")
                return int(contract),offer
            except Exception as exc:
                last_error=exc
                message=str(exc).lower()
                print(f"Offer {offer_id} rental failed: {exc}",flush=True)
                excluded_ids.add(offer_id)
                if "no_such_ask" in message or "not available" in message or "404/3603" in message:
                    # The snapshot is stale. Fetch a new offer list before the next attempt.
                    refresh_now=True
                    break
        if search_no < max_searches:
            time.sleep(min(2*search_no,8) if refresh_now else 1)
    if last_error:
        raise RuntimeError(f"No eligible offer could be rented after {attempts} attempts and "
                           f"{max_searches} fresh searches. Last error: {last_error}") from last_error
    raise RuntimeError(f"No eligible offer could be rented after {max_searches} fresh searches.")


def remote_training_script(epochs,samples):
    return f"""#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
cd /workspace/neo
mkdir -p artifacts datasets logs
echo '=== GPU preflight ==='
nvidia-smi
echo '=== Reuse CUDA PyTorch from the prebuilt Vast PyTorch image ==='
python3 -m venv --system-site-packages /workspace/neo/.venv
source /workspace/neo/.venv/bin/activate
command -v python
python --version
python -c "import torch; assert torch.cuda.is_available(), 'Preinstalled CUDA PyTorch is missing or cannot see the GPU'; print('CUDA PyTorch ready:', torch.__version__, torch.version.cuda); print('GPU:', torch.cuda.get_device_name(0))"
python -m pip install --no-deps -e .
test -s datasets/combined_v1/train.npz
test -s datasets/combined_v1/val.npz

echo '=== Train baseline on CUDA; save best and latest checkpoints ==='
python training/train_translator.py --data datasets/combined_v1 --epochs {epochs} --samples {samples} --batch-size 64 --workers 4 --device cuda --patience 12 --min-delta 0.0001 --seed 20261111 --output artifacts/neo_fastframe_motion.pt 2>&1 | tee logs/gpu_training.log

echo '=== Verify outputs ==='
test -s artifacts/neo_fastframe_motion.pt
test -s artifacts/neo_fastframe_motion.latest.pt
test -s artifacts/neo_fastframe_motion.metrics.json
python - <<'PY'
import json
from pathlib import Path
m=json.loads(Path('artifacts/neo_fastframe_motion.metrics.json').read_text())
assert m['epochs_completed']>=1 and m['best_epoch']>=1 and m['train_samples']>100
print(json.dumps({{k:m[k] for k in ('device','epochs_requested','epochs_completed','best_epoch','best_validation_score','train_samples','validation_samples','elapsed_seconds','model_scope')}},indent=2))
PY
nvidia-smi
"""

def run_deepseek_advisor(error, context=""):
    print("=== Requesting DeepSeek failure analysis ===", flush=True)
    command=[sys.executable, "scripts/deepseek_agent.py", "--error", str(error),
             "--context", context]
    try:
        subprocess.run(command, env=os.environ.copy(), text=True, timeout=50, check=False)
    except Exception as exc:
        print(f"DeepSeek advisor launch failed: {exc}", file=sys.stderr, flush=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--budget-usd",type=parse_usd,required=True)
    p.add_argument("--max-hourly-usd",type=parse_usd,required=True)
    p.add_argument("--max-hours",type=float,required=True)
    p.add_argument("--epochs",type=int,default=80)
    p.add_argument("--samples",type=int,default=12000)
    p.add_argument("--gpu-order",default="RTX 3090,RTX 4090")
    p.add_argument("--disk-gb",type=int,default=100)
    p.add_argument("--output",type=Path,default=Path("vast-training-output"))
    a=p.parse_args()
    key=os.environ.get("VAST_API_KEY","").strip()
    if not key:
        raise SystemExit("VAST_API_KEY missing. Add it in GitHub repository Settings > Secrets and variables > Actions.")
    if a.budget_usd<=0 or a.max_hourly_usd<=0: raise SystemExit("Budget and hourly cap must be positive.")
    if not 0.25<=a.max_hours<=3.0: raise SystemExit("max-hours must be between 0.25 and 3.0.")
    if not 1<=a.epochs<=200: raise SystemExit("epochs must be between 1 and 200.")
    if not 1000<=a.samples<=50000: raise SystemExit("samples must be between 1,000 and 50,000.")
    if not 80<=a.disk_gb<=250: raise SystemExit("disk-gb must be between 80 and 250.")
    if not shutil.which("ssh") or not shutil.which("ssh-keygen"): raise SystemExit("OpenSSH utilities are required.")

    api=VastAPI(key)
    # Offer IDs can expire between search and rental; rent_with_refresh re-searches safely.
    print("GPU policy: verified on-demand RTX 3090/4090, >=24GB VRAM, >=99% reliability; "
          f"hourly cap={a.max_hourly_usd:.2f}, total budget={a.budget_usd:.2f}, "
          f"runtime cap={a.max_hours:.2f}h",flush=True)

    a.output.mkdir(parents=True,exist_ok=True)
    temp=Path(tempfile.mkdtemp(prefix="neo-vast-ssh-"))
    key_path=temp/"vast_training_ed25519"
    env=os.environ.copy()
    instance_id=None; ssh_base=None; agent_started=False; training_error=None; cleanup_error=None
    selected=None; start=time.monotonic(); deadline=start+a.max_hours*3600
    try:
        start_agent(env); agent_started=True
        run_local(["ssh-keygen","-q","-t","ed25519","-N","","-f",str(key_path)],env=env)
        public_key=Path(str(key_path)+".pub").read_text(encoding="utf-8").strip()
        run_local(["vastai","create","ssh-key",public_key],env=env,timeout=60)

        instance_id,selected=rent_with_refresh(
            api,["RTX 3090","RTX 4090"],a.gpu_order,a.max_hourly_usd,
            a.budget_usd,a.max_hours,a.disk_gb,
            f"NEO-FastFrame-Training-{os.environ.get('GITHUB_RUN_ID','manual')}",
            max_searches=5,offers_per_search=8)

        print(f"Vast instance created: {instance_id}",flush=True)
        ready_until=min(deadline,time.monotonic()+20*60)
        while time.monotonic()<ready_until:
            state=api.get_instance(instance_id)
            status=str(state.get("actual_status",state.get("status",""))).lower()
            print(f"Vast instance {instance_id} status: {status or 'unknown'}",flush=True)
            if status=="running": break
            if status in {"dead","error","exited","deleted","stopped"}:
                raise RuntimeError(f"Vast instance entered terminal state: {status}")
            time.sleep(15)
        else: raise TimeoutError("Vast instance did not reach running state in 20 minutes.")

        ssh_until=min(deadline,time.monotonic()+10*60)
        while time.monotonic()<ssh_until:
            value=run_local(["vastai","ssh-url",str(instance_id)],env=env,timeout=30)
            tokens=shlex.split(value)
            ssh_options=["-i",str(key_path),"-o","IdentitiesOnly=yes",
                "-o","StrictHostKeyChecking=no","-o","UserKnownHostsFile=/dev/null",
                "-o","ConnectTimeout=15"]
            if value.startswith("ssh://"):
                parsed=urlparse(value)
                if not parsed.hostname or not parsed.port:
                    raise RuntimeError(f"Vast returned an invalid SSH URL: {value!r}")
                destination=f"{parsed.username or 'root'}@{parsed.hostname}"
                ssh_base=["ssh",*ssh_options,"-p",str(parsed.port),destination]
            else:
                if tokens and tokens[0]=="ssh": tokens=tokens[1:]
                ssh_base=["ssh",*ssh_options]+tokens if tokens else None
            if ssh_base:
                try:
                    run_local(ssh_base+["nvidia-smi"],env=env,timeout=30)
                    break
                except Exception: ssh_base=None
            time.sleep(10)
        if ssh_base is None: raise TimeoutError("Vast reports running but SSH/NVIDIA preflight did not become ready.")
        rate=as_float(selected,"dph_total",as_float(selected,"dph"))
        print(f"Selected GPU: {selected.get('gpu_name')} at {rate:.4f} USD/hour",flush=True)

        # Transfer only the checked-out source archive; no API key or PAT reaches the GPU host.
        source=subprocess.Popen(["git","archive","--format=tar.gz","HEAD"],stdout=subprocess.PIPE)
        unpack="mkdir -p /workspace/neo && tar -xzf - -C /workspace/neo"
        receiver=subprocess.Popen(ssh_base+[f"bash -lc {shlex.quote(unpack)}"],stdin=source.stdout)
        source.stdout.close()
        receive_rc=receiver.wait(timeout=300); source_rc=source.wait(timeout=60)
        if receive_rc or source_rc: raise RuntimeError("Source transfer to Vast worker failed.")

        # Blender captures and datasets are prepared on the free GitHub runner before rental.
        input_dir=Path("vast-training-input")
        if not (input_dir/"combined_v1"/"train.npz").is_file() or not (input_dir/"combined_v1"/"val.npz").is_file():
            raise RuntimeError("Prepared Blender/synthetic dataset is missing; refusing to spend GPU time on data preparation.")
        dataset_tar=subprocess.Popen(["tar","-czf","-","-C",str(input_dir),"."],stdout=subprocess.PIPE)
        dataset_unpack="mkdir -p /workspace/neo/datasets && tar -xzf - -C /workspace/neo/datasets"
        dataset_receiver=subprocess.Popen(ssh_base+[f"bash -lc {shlex.quote(dataset_unpack)}"],stdin=dataset_tar.stdout)
        dataset_tar.stdout.close()
        dataset_receive_rc=dataset_receiver.wait(timeout=300); dataset_tar_rc=dataset_tar.wait(timeout=60)
        if dataset_receive_rc or dataset_tar_rc:
            raise RuntimeError("Prepared training dataset transfer to Vast worker failed.")

        script=remote_training_script(a.epochs,a.samples)
        remote="bash -lc "+shlex.quote(script)
        remaining=max(60,int(deadline-time.monotonic()))
        try:
            result=subprocess.run(ssh_base+[remote],env=env,text=True,timeout=remaining,check=False)
            if result.returncode: training_error=RuntimeError(f"Remote GPU training exited with code {result.returncode}.")
        except subprocess.TimeoutExpired:
            training_error=TimeoutError("Maximum configured rental time was reached during training.")
            try:
                stop="pkill -TERM -f 'training/train_translator.py' || true"
                subprocess.run(ssh_base+["bash -lc "+shlex.quote(stop)],env=env,timeout=30,check=False)
            except Exception: pass

        archive_path=a.output/"vast-worker-output.tar.gz"
        pack="cd /workspace/neo && mkdir -p artifacts datasets logs && tar -czf - artifacts datasets logs"
        try:
            with archive_path.open("wb") as handle:
                subprocess.run(ssh_base+[pack],env=env,stdout=handle,
                    stderr=subprocess.PIPE,timeout=180,check=True)
            with tarfile.open(archive_path,"r:gz") as bundle:
                members=bundle.getmembers()
                for member in members:
                    target=Path(member.name)
                    if target.is_absolute() or ".." in target.parts:
                        raise RuntimeError("Unsafe path in worker result archive")
                bundle.extractall(a.output,members=members)
            archive_path.unlink(missing_ok=True)
        except Exception as exc:
            print(f"Warning: could not collect all worker outputs: {exc}",flush=True)

        info={"vast_instance_id":instance_id,"gpu_name":selected.get("gpu_name"),
            "offer_id":selected.get("id"),"hourly_rate_usd":rate,
            "max_budget_usd":a.budget_usd,"max_hourly_usd":a.max_hourly_usd,
            "max_runtime_hours":a.max_hours,"epochs_requested":a.epochs,
            "synthetic_samples_requested":a.samples,"source_branch":os.environ.get("GITHUB_REF_NAME"),
            "source_commit":run_local(["git","rev-parse","HEAD"]),"elapsed_seconds":time.monotonic()-start,
            "training_error":str(training_error) if training_error else None}
        (a.output/"vast_run_info.json").write_text(json.dumps(info,indent=2),encoding="utf-8")
        print(json.dumps(info,indent=2),flush=True)
        if training_error:
            run_deepseek_advisor(
                training_error,
                "Vast image: vastai/pytorch. The previous observed failure was "
                "'python: command not found' after NVIDIA-SMI detected an RTX 3090. "
                "The current run installs python3/python-is-python3 and verifies "
                "CUDA-enabled PyTorch before Blender capture and CUDA training. "
                f"Budget cap USD {a.budget_usd}; hourly cap USD {a.max_hourly_usd}; "
                f"runtime cap hours {a.max_hours}. Do not suggest exceeding these caps."
            )
            raise training_error
        if not (a.output/"artifacts/neo_fastframe_motion.pt").is_file():
            raise RuntimeError("Training ended without producing the best checkpoint.")
    finally:
        if instance_id is not None:
            try:
                print(f"Destroying Vast instance {instance_id}...",flush=True)
                api.destroy(instance_id)
                print("Vast destroy request accepted.",flush=True)
            except Exception as exc:
                cleanup_error=exc
                print(f"CRITICAL: could not destroy Vast instance {instance_id}: {exc}. Destroy it manually in Vast console.",
                      file=sys.stderr,flush=True)
        if agent_started:
            subprocess.run(["ssh-agent","-k"],env=env,check=False,
                           stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        shutil.rmtree(temp,ignore_errors=True)
    if cleanup_error:
        raise RuntimeError(f"Training completed but Vast cleanup failed: {cleanup_error}")

if __name__=="__main__": main()
