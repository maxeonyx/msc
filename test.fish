#!/usr/bin/fish

source env/bin/activate.fish

if test (which randomname)
    set PREV_BRANCH_NAME (git symbolic-ref --short HEAD)
    set -x RUN_NAME (randomname get)

    function fish_title
        echo "$RUN_NAME (train)"
    end
    
    # hide useless tensorflow logs
    set -x TF_CPP_MIN_LOG_LEVEL 1

    # fix for CUDA on archlinux
    set -x XLA_FLAGS --xla_gpu_cuda_data_dir=/opt/cuda/
    set -x TF_XLA_FLAGS --tf_xla_auto_jit=2 --tf_xla_cpu_global_jit

    mkdir -p _runs/$RUN_NAME
    # create a new commit with the current state of the working directory on a new branch
    git checkout -b run/$RUN_NAME
    git commit -a -m "Commit of src at run $RUN_NAME"
    git checkout --detach HEAD
    git reset --soft $PREV_BRANCH_NAME
    git checkout $PREV_BRANCH_NAME

    python train.py
    
    if test $status -eq 0
        function fish_title
            echo "$RUN_NAME (_anims)"
        end
        python create_animation.py
        if test $status -eq 0
            echo Run $RUN_NAME complete!
        else
            echo "$RUN_NAME _anims failed."
        end
    else
        echo "$RUN_NAME train failed."
    end

else
    echo "randomname not installed. run pip install"
end
