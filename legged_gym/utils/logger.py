# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import os
import time
from datetime import datetime


class Logger:
    """Simple logger for training metrics"""
    
    def __init__(self, log_dir):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        self.log_file = os.path.join(log_dir, "training.log")
        
        # Initialize log file
        with open(self.log_file, 'w') as f:
            f.write(f"Training started at {datetime.now()}\n")
            f.write("="*50 + "\n")
    
    def log(self, message):
        """Log a message with timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        
        # Print to console
        print(log_entry.strip())
        
        # Write to file
        with open(self.log_file, 'a') as f:
            f.write(log_entry)
    
    def log_metrics(self, iteration, metrics_dict):
        """Log training metrics"""
        message = f"Iteration {iteration}: "
        for key, value in metrics_dict.items():
            message += f"{key}={value:.4f} "
        self.log(message) 