# Stub for single-GPU inference on Windows where megatron-core cannot be built.
# All parallel-state queries return single-rank / uninitialized values.

_initialized = False
sequence_parallel = False


def is_initialized():
    return _initialized

def initialize_model_parallel(*args, **kwargs):
    pass

def destroy_model_parallel():
    pass

def get_context_parallel_group():
    return None

def get_context_parallel_rank():
    return 0

def get_context_parallel_world_size():
    return 1

def get_data_parallel_group(with_context_parallel=False):
    return None

def get_data_parallel_rank(with_context_parallel=False):
    return 0

def get_data_parallel_world_size(with_context_parallel=False):
    return 1

def get_tensor_model_parallel_rank():
    return 0

def get_tensor_model_parallel_world_size():
    return 1

def get_pipeline_model_parallel_rank():
    return 0

def get_pipeline_model_parallel_world_size():
    return 1
