srun -p h100nvl \
    --account=cas_interactive \
    --gres=gpu:h100:1 \
    -t 10:00:00 \
    --pty /bin/bash
