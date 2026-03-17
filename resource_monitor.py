# resource_monitor.py

import time
import psutil
import os
import GPUtil
import torch

class SystemMonitor:
    def __init__(self):
        self.prev_time = time.time()
        self.fps = 0
        
        # 1. Grab the specific Process ID (PID) of this Python application
        self.process = psutil.Process(os.getpid())
        
        # Initialize the CPU percent counter (the first call usually returns 0)
        self.process.cpu_percent()

    def get_metrics(self):
        """Calculates and returns a dictionary of resources used by THIS app."""
        # 1. Calculate FPS
        current_time = time.time()
        self.fps = 1 / ((current_time - self.prev_time) + 0.0001) 
        self.prev_time = current_time

        # 2. App-Specific CPU and RAM
        # Divide by cpu_count() to match Windows Task Manager scaling (0-100% of total system)
        app_cpu_usage = self.process.cpu_percent() / psutil.cpu_count()
        
        # RSS (Resident Set Size) is the actual physical RAM used by this process
        app_ram_gb = self.process.memory_info().rss / (1024 ** 3)

        # 3. GPU Metrics (Hardware level)
        gpus = GPUtil.getGPUs()
        gpu_load = 0
        gpu_vram_total = 0
        
        if len(gpus) > 0:
            gpu = gpus[0]
            gpu_load = gpu.load * 100  
            gpu_vram_total = gpu.memoryUsed  

        # 4. PyTorch Specific VRAM (App level)
        # memory_reserved() shows exactly what this PyTorch app has claimed from the OS
        app_vram_reserved = torch.cuda.memory_reserved() / (1024 ** 2)

        return {
            "FPS": f"{self.fps:.1f}",
            "App CPU": f"{app_cpu_usage:.1f}%",
            "App RAM": f"{app_ram_gb:.2f} GB",
            "GPU Load (Total)": f"{gpu_load:.1f}%",
            "VRAM (Total)": f"{gpu_vram_total:.0f} MB",
            "App VRAM": f"{app_vram_reserved:.1f} MB"
        }