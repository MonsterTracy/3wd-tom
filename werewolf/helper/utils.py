import subprocess


role_en2cn = {
    "Werewolf": "狼人",
    "Villager": "村民",
    "Seer": "预言家",
    "Guard": "守卫",
    "Witch": "女巫",
}

role_cn2en = {
    "狼人": "Werewolf",
    "村民": "Villager",
    "预言家": "Seer",
    "守卫": "Guard",
    "女巫": "Witch",
}


def get_gpu_memory_map():
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used,memory.total',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    gpu_memory = [x.split(',') for x in result.strip().split('\n')]
    gpu_memory_map = {
        i: {'used': int(memory_used), 'total': int(memory_total)}
        for i, (memory_used, memory_total) in enumerate(gpu_memory)
    }
    return gpu_memory_map

def get_available_devices(threshold=50000):
    try:
        import torch
    except ImportError:
        return "cpu"

    device = "auto"
    if torch.cuda.is_available():
        n_gpu = torch.cuda.device_count()
        if n_gpu == 1:
            device = "cuda:0"
        else:
            gpu_memory_map = get_gpu_memory_map()
            for i, gpu_status in gpu_memory_map.items():
                if gpu_status["used"] < threshold:
                    device = f"cuda:{i}"
                    break
    else:
        device = "cpu"
    return device
