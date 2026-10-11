python3 -m venv /workspace/venv_lora
/workspace/venv_lora/bin/pip install -U pip
rm -f /tmp/lora_install.log
cd /tmp
/workspace/venv_lora/bin/pip install --dry-run --ignore-installed --report /tmp/lora_report.json unsloth trl peft datasets 2>/dev/null | tail -1 || true
python3 - <<'PY'
import json
j = json.load(open('/tmp/lora_report.json'))
urls = [i['download_info']['url'] for i in j.get('install', []) if i.get('download_info')]
open('/tmp/lora_urls.txt', 'w').write('\n'.join(urls))
print('lora wheels:', len(urls))
PY
mkdir -p /tmp/lora_wheels
cd /tmp/lora_wheels
aria2c --input-file=/tmp/lora_urls.txt -j 5 -x 12 -s 12 -c --console-log-level=error --dir=/tmp/lora_wheels >/dev/null 2>&1 || true
/workspace/venv_lora/bin/pip install --no-index --find-links /tmp/lora_wheels unsloth trl peft datasets > /tmp/lora_install.log 2>&1
echo INSTALL_DONE