# Using GPUs
If a TensorFlow operation has both CPU and GPU implementations, the GPU devices will be given priority when the operation is assigned to a device. For example, matmul has both CPU and GPU kernels. On a system with devices cpu:0 and gpu:0, gpu:0 will be selected to run matmul.

## Reference
* https://www.tensorflow.org/tutorials/using_gpu

## Logging Device placement
To find out which devices your operations and tensors are assigned to, create the session with log_device_placement configuration option set to True.
```
$ python log_device_placement.py 

MatMul: /job:localhost/replica:0/task:0/gpu:0
I tensorflow/core/common_runtime/simple_placer.cc:819] MatMul: /job:localhost/replica:0/task:0/gpu:0
b: /job:localhost/replica:0/task:0/gpu:0
I tensorflow/core/common_runtime/simple_placer.cc:819] b: /job:localhost/replica:0/task:0/gpu:0
a: /job:localhost/replica:0/task:0/gpu:0
I tensorflow/core/common_runtime/simple_placer.cc:819] a: /job:localhost/replica:0/task:0/gpu:0
[[ 22.  28.]
 [ 49.  64.]]
```

## Manual device placement
If you would like a particular operation to run on a device of your choice instead of what's automatically selected for you, you can use with tf.device to create a device context such that all the operations within that context will have the same device assignment.
```
$ python manual_device_placement.py 

MatMul: /job:localhost/replica:0/task:0/cpu:0
I tensorflow/core/common_runtime/simple_placer.cc:819] MatMul: /job:localhost/replica:0/task:0/cpu:0
b: /job:localhost/replica:0/task:0/cpu:0
I tensorflow/core/common_runtime/simple_placer.cc:819] b: /job:localhost/replica:0/task:0/cpu:0
a: /job:localhost/replica:0/task:0/cpu:0
I tensorflow/core/common_runtime/simple_placer.cc:819] a: /job:localhost/replica:0/task:0/cpu:0
[[ 22.  28.]
 [ 49.  64.]]
```

## Allowing GPU memory growth
By default, TensorFlow maps nearly all of the GPU memory of all GPUs (subject to CUDA_VISIBLE_DEVICES) visible to the process. This is done to more efficiently use the relatively precious GPU memory resources on the devices by reducing memory fragmentation.

In some cases it is desirable for the process to only allocate a subset of the available memory, or to only grow the memory usage as is needed by the process. TensorFlow provides two Config options on the Session to control this.

The first is the allow_growth option, which attempts to allocate only as much GPU memory based on runtime allocations: it starts out allocating very little memory, and as Sessions get run and more GPU memory is needed, we extend the GPU memory region needed by the TensorFlow process. Note that we do not release memory, since that can lead to even worse memory fragmentation. To turn this option on, set the option in the ConfigProto by:
```
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config, ...)
```
The second method is the per_process_gpu_memory_fraction option, which determines the fraction of the overall amount of memory that each visible GPU should be allocated. For example, you can tell TensorFlow to only allocate 40% of the total memory of each GPU by:
```
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
session = tf.Session(config=config, ...)
```

## Using a single GPU on a multi-GPU system
```
$ python using_single_gpu_from_multi_gpu.py 

MatMul: /job:localhost/replica:0/task:0/gpu:1
I tensorflow/core/common_runtime/simple_placer.cc:819] MatMul: /job:localhost/replica:0/task:0/gpu:1
b: /job:localhost/replica:0/task:0/gpu:1
I tensorflow/core/common_runtime/simple_placer.cc:819] b: /job:localhost/replica:0/task:0/gpu:1
a: /job:localhost/replica:0/task:0/gpu:1
I tensorflow/core/common_runtime/simple_placer.cc:819] a: /job:localhost/replica:0/task:0/gpu:1
[[ 22.  28.]
 [ 49.  64.]]
```

## Using multiple GPUs
```
$ python using_multiple_gpus.py 

MatMul_1: /job:localhost/replica:0/task:0/gpu:1
I tensorflow/core/common_runtime/simple_placer.cc:819] MatMul_1: /job:localhost/replica:0/task:0/gpu:1
MatMul: /job:localhost/replica:0/task:0/gpu:0
I tensorflow/core/common_runtime/simple_placer.cc:819] MatMul: /job:localhost/replica:0/task:0/gpu:0
AddN: /job:localhost/replica:0/task:0/cpu:0
I tensorflow/core/common_runtime/simple_placer.cc:819] AddN: /job:localhost/replica:0/task:0/cpu:0
Const_3: /job:localhost/replica:0/task:0/gpu:1
I tensorflow/core/common_runtime/simple_placer.cc:819] Const_3: /job:localhost/replica:0/task:0/gpu:1
Const_2: /job:localhost/replica:0/task:0/gpu:1
I tensorflow/core/common_runtime/simple_placer.cc:819] Const_2: /job:localhost/replica:0/task:0/gpu:1
Const_1: /job:localhost/replica:0/task:0/gpu:0
I tensorflow/core/common_runtime/simple_placer.cc:819] Const_1: /job:localhost/replica:0/task:0/gpu:0
Const: /job:localhost/replica:0/task:0/gpu:0
I tensorflow/core/common_runtime/simple_placer.cc:819] Const: /job:localhost/replica:0/task:0/gpu:0
[[  44.   56.]
 [  98.  128.]]
```
