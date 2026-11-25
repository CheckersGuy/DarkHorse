#!/bin/bash
# after the 150-epoch run: install the policy/value loader with epoch rotation
# (Rust PolicyDataLoader rotate=true); must be done before the chain's final training
cd /home/leaguerobin/DarkHorse/Training
while pgrep -f "[r]un exp_L150_cos" >/dev/null; do sleep 30; done
/home/leaguerobin/training/bin/pip install --force-reinstall --no-deps Rust/wheels/movegen-0.1.0-cp312-cp312-manylinux_2_34_x86_64.whl > rotate_install.log 2>&1
/home/leaguerobin/training/bin/python -c "import string_sum; string_sum.PolicyBatchProvider('/mnt/c/TrainingData/reservoir-full.val.children.samples', 1000, 100, True, True); print('rotation loader installed')" >> rotate_install.log 2>&1
