pkill -f 'continuous_learne[r]' 2>/dev/null || true
sleep 2
rm -f /workspace/AGI/logs/continuous/checkpoint.json
printf '{"started_at":"%s"}\n' "$(date -u +%Y-%m-%dT%H:%M:%S)" > /workspace/AGI/logs/continuous/session.json
if [ -f /workspace/AGI/logs/training_queue.jsonl ]; then
  mv /workspace/AGI/logs/training_queue.jsonl /workspace/AGI/logs/training_queue_archive_$(date -u +%Y%m%d_%H%M%S).jsonl
fi
cd /workspace/AGI/scripts
setsid /opt/venv/bin/python continuous_learner.py --cycles 0 --tick 2.5 \
  --seed-task "What is 2+2?" \
  --seed-task "What is the capital of France?" \
  > /workspace/AGI/logs/learner.log 2>&1 < /dev/null &
echo "learner restarted"
