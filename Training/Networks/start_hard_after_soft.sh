#!/bin/bash
# Waits for the soft-label run (pid 209401) to finish, then starts the hard-label run.
cd /home/leaguerobin/DarkHorse/Training
while kill -0 209401 2>/dev/null; do sleep 30; done
echo "soft run finished at $(date), starting hard run" >> Networks/policy256_hard_train.log
exec /home/leaguerobin/training/bin/python train_policy.py --label-mode hard >> Networks/policy256_hard_train.log 2>&1 < /dev/null
