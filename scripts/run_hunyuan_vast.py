#!/usr/bin/env python3
"""Budget-capped Vast runner for HunyuanVideo 1.5 T2V."""
from __future__ import annotations
import argparse, json, os, re, shlex, shutil, subprocess, sys, tempfile, time, traceback
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
    deadline=float(os.environ.get("NEO_GPU_DEADLINE_MONOTONIC", "0") or 0)
    if deadline <= 0:
        deadline=time.monotonic()+a.max_hours*3600
        os.environ["NEO_GPU_DEADLINE_MONOTONIC"]=str(deadline)
    def remaining_seconds(stage, cap=None):
        left=deadline-time.monotonic()
        if left<=0:
            raise TimeoutError(f"Total GPU rental time cap reached before {stage}.")
        return min(left,cap) if cap is not None else left
    key=os.environ.get("VAST_API_KEY","").strip()
    if not key: raise SystemExit("VAST_API_KEY secret is missing.")
    s=requests.Session(); s.headers.update({"Authorization":f"Bearer {key}","Content-Type":"application/json"})
    def api(method,path,payload=None,timeout=45):
        r=s.request(method,API+path,json=payload,timeout=timeout)
        if not r.ok: raise RuntimeError(f"Vast API {method} {path}: HTTP {r.status_code} {r.text[:500]}")
        return r.json() if r.text.strip() else {}

    blacklist_path=Path("gpu-host-blacklist.json")
    try:
        blacklist_data=json.loads(blacklist_path.read_text()) if blacklist_path.exists() else {"hosts":[]}
    except Exception as exc:
        raise RuntimeError(f"Cannot read GPU host blacklist safely: {exc}")
    blocked=blacklist_data.get("hosts",[])
    def host_identity(o):
        geo=o.get("geolocation") or o.get("location") or {}
        if not isinstance(geo,dict): geo={"raw":str(geo)}
        machine=str(o.get("machine_id") or "")
        host=str(o.get("host_id") or "")
        parts=[machine,host,str(geo.get("country","")),str(geo.get("region","")),str(geo.get("city","")),
               str(o.get("gpu_name","")),str(o.get("cpu_name","")),str(o.get("cpu_arch",""))]
        return {"machine_id":machine,"host_id":host,"geolocation":geo,
                "gpu_name":str(o.get("gpu_name","")),"cpu_name":str(o.get("cpu_name","")),
                "cpu_arch":str(o.get("cpu_arch","")),"fingerprint":"|".join(parts)}
    def is_blacklisted(o):
        ident=host_identity(o)
        for entry in blocked:
            if ident["machine_id"] and entry.get("machine_id")==ident["machine_id"]: return True
            if ident["host_id"] and entry.get("host_id")==ident["host_id"]: return True
            if ident["fingerprint"] and entry.get("fingerprint")==ident["fingerprint"]: return True
        return False
    def persist_blacklist(offer, measured_mbps):
        ident=host_identity(offer)
        if not ident["machine_id"] and not ident["host_id"]:
            raise RuntimeError("Cannot persistently blacklist slow host: Vast offer omitted machine_id and host_id.")
        entry={**ident,"reason":"download speed below 65 MB/s","measured_MBps":round(measured_mbps,2),
               "threshold_MBps":65,"blacklisted_at_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())}
        if not any((entry["machine_id"] and x.get("machine_id")==entry["machine_id"]) or
                   (entry["host_id"] and x.get("host_id")==entry["host_id"]) for x in blocked):
            blocked.append(entry)
        blacklist_data["hosts"]=blocked
        token=os.environ.get("GITHUB_TOKEN","").strip()
        repo=os.environ.get("GITHUB_REPOSITORY","")
        branch=os.environ.get("GITHUB_REF_NAME","hunyuan-t2v-16gb")
        if not token or not repo:
            raise RuntimeError("GITHUB_TOKEN/GITHUB_REPOSITORY missing; cannot persist slow-host blacklist.")
        gh=requests.Session()
        gh.headers.update({"Authorization":f"Bearer {token}","Accept":"application/vnd.github+json","X-GitHub-Api-Version":"2022-11-28"})
        url=f"https://api.github.com/repos/{repo}/contents/gpu-host-blacklist.json"
        current=gh.get(url,params={"ref":branch},timeout=30)
        sha=None
        if current.status_code==200: sha=current.json().get("sha")
        elif current.status_code!=404: raise RuntimeError(f"Blacklist read failed: HTTP {current.status_code} {current.text[:300]}")
        import base64
        payload={"message":f"Blacklist slow Vast GPU host {ident['machine_id'] or ident['host_id']}",
                 "content":base64.b64encode((json.dumps(blacklist_data,indent=2)+"\\n").encode()).decode(),
                 "branch":branch}
        if sha: payload["sha"]=sha
        saved=gh.put(url,json=payload,timeout=30)
        if not saved.ok: raise RuntimeError(f"Blacklist persistence failed: HTTP {saved.status_code} {saved.text[:500]}")
        print(f"GPU_HOST_BLACKLISTED machine_id={ident['machine_id']} host_id={ident['host_id']} location={json.dumps(ident['geolocation'],sort_keys=True)} gpu={ident['gpu_name']} cpu={ident['cpu_name']} speed_MBps={measured_mbps:.2f}",flush=True)
    offers=api("POST","/bundles/",{
        "gpu_name":{"in":["RTX 5060 Ti","RTX 5070 Ti","RTX 4070 Ti SUPER","RTX 4080","RTX 5080","RTX 4060 Ti"]},
        "num_gpus":{"eq":1},"gpu_ram":{"gte":14},"cpu_ram":{"gte":48},
        "disk_space":{"gte":a.disk_gb},"reliability":{"gte":0.99},
        "verified":{"eq":True},"rentable":{"eq":True},"type":"ondemand","limit":50
    }).get("offers",[])
    candidates=[]
    skipped_blacklisted=0
    for o in offers:
        name=str(o.get("gpu_name","")).upper().replace("_"," ")
        rate=float(o.get("dph_total",o.get("dph",0)) or 0)
        if not any(n in name for n in ("RTX 5060 TI","RTX 5070 TI","RTX 4070 TI SUPER","RTX 4080","RTX 5080","RTX 4060 TI")): continue
        if rate<=0 or rate>a.max_hourly_usd or rate*a.max_hours*1.25>a.budget_usd: continue
        if is_blacklisted(o):
            skipped_blacklisted+=1
            continue
        candidates.append((rate,-float(o.get("reliability",0) or 0),o))
    print(f"GPU_HOST_BLACKLIST_LOADED={len(blocked)} matching_offers_skipped={skipped_blacklisted}",flush=True)
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
        run(["vastai","set","api-key",key],env=env)
        kp=temp/"key"; run(["ssh-keygen","-q","-t","ed25519","-N","","-f",str(kp)])
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
        print(f"GPU_RUNTIME_DEADLINE_SECONDS={max(0,deadline-time.monotonic()):.0f} remaining (shared across retries, boot, SSH, setup, downloads, generation and transfer)",flush=True)
        def get_instance():
            d=api("GET",f"/instances/{instance}/",timeout=min(15,remaining_seconds("instance status check"))); x=d.get("instances",d)
            if isinstance(x,list): return next((z for z in x if int(z.get("id",-1))==int(instance)),{})
            return x if isinstance(x,dict) else {}
        ready_deadline=min(deadline,time.monotonic()+1200)
        while time.monotonic()<ready_deadline:
            st=get_instance(); status=str(st.get("actual_status",st.get("status",""))).lower()
            print(f"INSTANCE_STATUS={status}",flush=True)
            if status=="running": break
            if status in ("dead","error","exited","deleted","stopped"): raise RuntimeError(f"Instance failed: {status}")
            time.sleep(min(15,remaining_seconds("GPU boot wait")))
        else: raise TimeoutError("GPU did not become running within the startup limit or the one-hour rental deadline.")
        ssh_base=None; ssh_deadline=min(deadline,time.monotonic()+600)
        while time.monotonic()<ssh_deadline:
            value=run(["vastai","ssh-url",str(instance)],env=env,capture_output=True,timeout=min(30,remaining_seconds("SSH endpoint discovery"))).stdout.strip()
            opts=["-i",str(kp),"-o","IdentitiesOnly=yes","-o","StrictHostKeyChecking=no","-o","UserKnownHostsFile=/dev/null","-o","ConnectTimeout=12"]
            if value.startswith("ssh://"):
                from urllib.parse import urlparse
                u=urlparse(value); ssh_base=["ssh",*opts,"-p",str(u.port),f"{u.username or 'root'}@{u.hostname}"]
            else:
                toks=shlex.split(value); toks=toks[1:] if toks and toks[0]=="ssh" else toks
                ssh_base=["ssh",*opts,*toks] if toks else None
            if ssh_base:
                probe=run(ssh_base+["nvidia-smi"],env=env,capture_output=True,check=False,timeout=min(25,remaining_seconds("SSH/NVIDIA preflight")))
                if probe.returncode==0: print(probe.stdout,flush=True); break
                ssh_base=None
            time.sleep(min(10,remaining_seconds("SSH availability wait")))
        if not ssh_base: raise TimeoutError("SSH/NVIDIA preflight never became ready before the rental deadline.")
        # Benchmark network before sending source code or downloading multi-GB models.
        # Four concurrent 32 MiB streams = 128 MiB total benchmark traffic.
        print("STARTING_PRE_DOWNLOAD_SPEED_TEST threshold_MBps=65 test_bytes=134217728",flush=True)
        speed_script = r"""python3 - <<'PY'
import concurrent.futures, time, urllib.request
URL='https://speed.cloudflare.com/__down?bytes=33554432'
def fetch(_):
    req=urllib.request.Request(URL,headers={'User-Agent':'NEO-Hunyuan-Network-Probe/1.0','Cache-Control':'no-cache'})
    start=time.monotonic(); total=0
    with urllib.request.urlopen(req,timeout=25) as response:
        while True:
            chunk=response.read(1024*1024)
            if not chunk: break
            total += len(chunk)
    return total, time.monotonic()-start
start=time.monotonic()
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    results=list(pool.map(fetch,range(4)))
elapsed=time.monotonic()-start
total=sum(x[0] for x in results)
mbps=(total/1048576)/elapsed if elapsed>0 else 0
print(f'PRE_DOWNLOAD_SPEED_TEST bytes={total} elapsed_seconds={elapsed:.3f} download_MBps={mbps:.2f}',flush=True)
if total < 120*1024*1024:
    raise SystemExit('SPEED_TEST_INCOMPLETE')
if mbps < 90:
    raise SystemExit(f'SLOW_GPU_HOST: {mbps:.2f} MB/s is below 90 MB/s')
PY"""
        speed_result=run(ssh_base+["bash -lc "+shlex.quote(speed_script)],env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,check=False,timeout=min(50,remaining_seconds("pre-download network speed test")))
        print(speed_result.stdout or "",flush=True)
        if speed_result.returncode:
            match=re.search(r"download_MBps=([0-9.]+)",speed_result.stdout or "")
            measured=float(match.group(1)) if match else 0.0
            if "SLOW_GPU_HOST" in (speed_result.stdout or ""):
                persist_blacklist(selected,measured)
                print(f"SLOW_HOST_RETRY_REQUIRED measured_MBps={measured:.2f} threshold_MBps=65",flush=True)
                raise RuntimeError(f"Host download speed {measured:.2f} MB/s is below 65 MB/s. Host blacklisted; trying another GPU host.")
            raise RuntimeError(f"Pre-download speed test failed or was incomplete (exit={speed_result.returncode}); no model download started.")
        source=subprocess.Popen(["git","archive","--format=tar.gz","HEAD"],stdout=subprocess.PIPE)
        recv=subprocess.Popen(ssh_base+["bash -lc "+shlex.quote("mkdir -p /workspace/neo && tar -xzf - -C /workspace/neo")],stdin=source.stdout)
        source.stdout.close()
        if recv.wait(timeout=min(300,remaining_seconds("source transfer")))!=0 or source.wait(timeout=min(60,remaining_seconds("source archive completion")))!=0: raise RuntimeError("Source transfer failed.")
        # Stream remote generation output directly into the Actions log. The old
        # capture_output=True buffered everything until the job ended.
        script = "cd /workspace/neo && chmod +x scripts/run_hunyuan_t2v_16gb.sh && " + "./scripts/run_hunyuan_t2v_16gb.sh " + shlex.quote(a.prompt)
        remote_cmd = (
            'LOG=/tmp/neo-hunyuan-generation.log; '
            '(' + script + ') > >(tee "$LOG") 2>&1 & PID=$!; '
            '(while kill -0 "$PID" 2>/dev/null; do '
            'echo "[GPU_TELEMETRY]"; '
            'nvidia-smi --query-gpu=name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader 2>&1; '
            'echo "[MODEL_DOWNLOAD_TELEMETRY]"; '
            'du -sb /workspace/neo-data/models/hunyuanvideo15/ckpts 2>/dev/null || true; '
            'df -h /workspace/neo-data 2>/dev/null | tail -n 1; '
            'IFACE=$(ip -o route show default 2>/dev/null | awk \'{print $5; exit}\'); '
            'if [ -n "$IFACE" ]; then RX=$(cat /sys/class/net/"$IFACE"/statistics/rx_bytes 2>/dev/null || echo 0); TX=$(cat /sys/class/net/"$IFACE"/statistics/tx_bytes 2>/dev/null || echo 0); '
            'if [ -n "$PREV_RX" ]; then DELTA=$((RX-PREV_RX)); RATE=$(awk -v d="$DELTA" \'BEGIN {printf "%.2f", d/20/1048576}\'); TOTAL_GB=$(awk -v r="$RX" \'BEGIN {printf "%.3f", r/1073741824}\'); echo "[NETWORK_SPEED] iface=$IFACE download_MBps=$RATE total_rx_GB=$TOTAL_GB"; '
            'else echo "[NETWORK_SPEED] iface=$IFACE measuring_first_interval"; fi; PREV_RX=$RX; fi; '
            'sleep 20; done) & MONITOR=$!; '
            'wait "$PID"; RC=$?; kill "$MONITOR" 2>/dev/null; '
            'wait "$MONITOR" 2>/dev/null; exit "$RC"'
        )
        remaining=remaining_seconds("remote dependency installation, model downloads and video generation")
        print(f"STARTING_HUNYUAN_GENERATION_WITH_LIVE_LOGS_AND_GPU_TELEMETRY remaining_seconds={remaining:.1f}",flush=True)
        result=run(ssh_base+["bash -lc "+shlex.quote(remote_cmd)],env=env,timeout=remaining,check=False)
        if result.returncode: raise RuntimeError(f"Hunyuan generation exited with code {result.returncode}")
        found=run(ssh_base+["bash -lc "+shlex.quote("find /workspace/neo-data/outputs/hunyuan -maxdepth 1 -type f -name '*.mp4' -print -quit")],env=env,capture_output=True,timeout=remaining_seconds("generated video lookup")).stdout.strip()
        if not found: raise RuntimeError("Generation completed but no MP4 was found.")
        local=out/"hunyuan_t2v.mp4"
        with local.open("wb") as f:
            cp=run(ssh_base+["cat "+shlex.quote(found)],env=env,stdout=f,stderr=subprocess.PIPE,timeout=remaining_seconds("generated video transfer"),check=False)
        if cp.returncode or local.stat().st_size==0: raise RuntimeError("Failed to download generated MP4.")
        print(f"VIDEO_OUTPUT={local} BYTES={local.stat().st_size}",flush=True)
    except Exception as exc:
        failure = traceback.format_exc()
        if isinstance(exc, subprocess.TimeoutExpired):
            for label, value in (("TIMEOUT_STDOUT", exc.stdout), ("TIMEOUT_STDERR", exc.stderr)):
                if value:
                    if isinstance(value, bytes): value=value.decode("utf-8", errors="replace")
                    print(f"{label}:\\n{value[-12000:]}", file=sys.stderr, flush=True)
        print("NEO_HUNYUAN_FAILURE\\n"+failure, file=sys.stderr, flush=True)
        # Stop paid compute before asking DeepSeek to diagnose the failure.
        slow_host_retry = "SLOW_HOST_RETRY_REQUIRED" in failure
        if instance is not None:
            try:
                print(f"DESTROYING_INSTANCE_BEFORE_DIAGNOSIS={instance}",flush=True)
                api("DELETE",f"/instances/{instance}/")
                print("DESTROY_REQUEST_ACCEPTED",flush=True)
                instance=None
            except Exception as destroy_error:
                detail=str(destroy_error).lower()
                if "no_such_instance" in detail or "http 404" in detail:
                    print(f"INSTANCE_ALREADY_GONE instance={instance}; Vast reports no active instance.",flush=True)
                    instance=None
                else:
                    print(f"CRITICAL_DESTROY_FAILURE instance={instance}: {destroy_error}",file=sys.stderr,flush=True)
        context = (
            f"Workflow: NEO HunyuanVideo 1.5 T2V\\nPrompt: {a.prompt}\\n"
            f"Budget USD: {a.budget_usd}; max hourly USD: {a.max_hourly_usd}; "
            f"max hours: {a.max_hours}\\nGPU instance: {instance or 'destroy requested'}\\n"
            "Policy: never raise budget or runtime caps. GPU must be destroyed on every failure. "
            "Give diagnosis and minimal fix; do not claim code was changed."
        )
        if os.environ.get("DEEPSEEK_API_KEY","").strip():
            advisor = run([sys.executable,"scripts/deepseek_agent.py","--error",failure,"--context",context],
                          capture_output=True,check=False,timeout=55)
            if advisor.stdout: print(advisor.stdout[-12000:],flush=True)
            if advisor.stderr: print(advisor.stderr[-4000:],file=sys.stderr,flush=True)
        else:
            print("DEEPSEEK_DIAGNOSIS_SKIPPED: DEEPSEEK_API_KEY GitHub Actions secret is not configured.",flush=True)
        if slow_host_retry and instance is None:
            left=deadline-time.monotonic()
            if left>180:
                print(f"RENTING_NEXT_GPU_HOST remaining_runtime_seconds={left:.0f}",flush=True)
                if agent:
                    run(["ssh-agent","-k"],env=env,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,check=False)
                    agent=False
                shutil.rmtree(temp,ignore_errors=True)
                os.execv(sys.executable,[sys.executable,*sys.argv])
            print(f"NO_RETRY_RUNTIME_REMAINING seconds={max(0,left):.0f}",flush=True)
        raise
    finally:
        if instance is not None:
            try:
                print(f"DESTROYING_INSTANCE={instance}",flush=True); api("DELETE",f"/instances/{instance}/")
                print("DESTROY_REQUEST_ACCEPTED",flush=True)
            except Exception as e:
                detail=str(e).lower()
                if "no_such_instance" in detail or "http 404" in detail:
                    print(f"INSTANCE_ALREADY_GONE instance={instance}; no active Vast contract remains.",flush=True)
                else:
                    print(f"CRITICAL_DESTROY_FAILURE instance={instance}: {e}",file=sys.stderr,flush=True)
        if agent: run(["ssh-agent","-k"],env=env,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,check=False)
        shutil.rmtree(temp,ignore_errors=True)

if __name__=="__main__": main()
