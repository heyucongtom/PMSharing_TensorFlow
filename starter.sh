bazel-bin/tensorflow/core/distributed_runtime/rpc/grpc_tensorflow_server \
    --cluster_spec='worker|worker0:2222;worker1:2222,ps|ps0:2222' --job_name=local --task_index=0 &
