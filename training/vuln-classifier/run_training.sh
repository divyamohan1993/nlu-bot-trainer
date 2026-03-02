#!/bin/bash
pkill -f "python3 train.py" 2>/dev/null
sleep 1
cd /opt/vuln-trainer
source venv/bin/activate
nohup python3 train.py --epochs 50 --batch-size 512 --lr 0.003 > training.log 2>&1 &
echo "STARTED PID=$!"
