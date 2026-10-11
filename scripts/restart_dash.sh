pkill -f 'dashboar[d]' 2>/dev/null || true
sleep 2
cd /workspace/AGI/scripts
setsid /opt/venv/bin/python dashboard.py --host 0.0.0.0 --port 8010 > /workspace/AGI/logs/dashboard.log 2>&1 < /dev/null &
echo "started"
sleep 7
curl -s -o /dev/null -w 'root=%{http_code}\n' http://localhost:8010/