# backup
cp ~/.docker/config.json ~/.docker/config.json.bak 2>/dev/null || true

# set credsStore to osxkeychain (create file if missing)
python3 - <<'PY'
import json, os, pathlib
p = pathlib.Path.home()/'.docker'/'config.json'
p.parent.mkdir(parents=True, exist_ok=True)
cfg = {}
if p.exists():
    try: cfg = json.load(open(p))
    except: cfg = {}
cfg["credsStore"] = "osxkeychain"
json.dump(cfg, open(p,'w'), indent=2)
print("Updated", p, "to use credsStore=osxkeychain")
PY
