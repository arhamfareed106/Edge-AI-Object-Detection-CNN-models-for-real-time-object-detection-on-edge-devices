"""
Hardware Monitor Module for Raspberry Pi Resource Tracking

Monitors CPU temperature, CPU usage, RAM usage, and estimates energy consumption
during model inference on Raspberry Pi 4.

Author: Edge AI Research Team
Date: 2024
"""

import logging
import time
from typing import Dict, Optional, List
from datetime import datetime
from pathlib import Path

import psutil

logger = logging.getLogger(__name__)


class HardwareMonitor:
    """
    Monitor Raspberry Pi hardware resources and energy consumption.
    
    Tracks CPU temperature, CPU utilization, memory usage, and provides
    energy consumption estimates for benchmarking purposes.
    """
    
    # Raspberry Pi 4 power consumption estimates (Watts)
    POWER_IDLE = 3.5       # Idle power
    POWER_CPU_50 = 5.0     # 50% CPU load
    POWER_CPU_100 = 6.5    # 100% CPU load
    POWER_GPU = 1.5        # Additional GPU/Camera power
    
    def __init__(self, log_interval: float = 0.1, log_file: Optional[str] = None):
        """
        Initialize hardware monitor.
        
        Args:
            log_interval: Time interval between measurements (seconds)
            log_file: Optional file path to save hardware logs
        """
        self.log_interval = log_interval
        self.monitoring = False
        self.measurements: List[Dict] = []
        self.start_time: Optional[float] = None
        
        # Set up file logging if specified
        if log_file:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)
            logger.addHandler(file_handler)
        
        logger.info("HardwareMonitor initialized with interval %.2fs", log_interval)
    
    def get_cpu_temperature(self) -> float:
        """
        Get CPU temperature in Celsius.
        
        Returns:
            CPU temperature in degrees Celsius, or 0.0 if unavailable
        """
        try:
            # Try Raspberry Pi specific method first
            import subprocess
            result = subprocess.run(
                ['vcgencmd', 'measure_temp'],
                capture_output=True,
                text=True,
                timeout=2
            )
            if result.returncode == 0:
                # Output format: temp=52.3'C
                temp_str = result.stdout.split('=')[1].split("'")[0]
                return float(temp_str)
        except (FileNotFoundError, subprocess.TimeoutExpired, IndexError):
            pass
        
        # Fallback to psutil sensors
        try:
            temps = psutil.sensors_temperatures()
            if 'coretemp' in temps:
                return temps['coretemp'][0].current
            elif 'cpu_thermal' in temps:
                return temps['cpu_thermal'][0].current
        except (AttributeError, KeyError):
            pass
        
        logger.warning("Temperature reading unavailable")
        return 0.0
    
    def get_cpu_usage(self) -> float:
        """
        Get current CPU usage percentage.
        
        Returns:
            CPU usage as percentage (0-100)
        """
        return psutil.cpu_percent(interval=0.01)
    
    def get_memory_usage(self) -> Dict[str, float]:
        """
        Get memory usage statistics.
        
        Returns:
            Dictionary with 'used_mb', 'total_mb', and 'percent' keys
        """
        memory = psutil.virtual_memory()
        return {
            'used_mb': memory.used / (1024 ** 2),
            'total_mb': memory.total / (1024 ** 2),
            'percent': memory.percent
        }
    
    def estimate_power_consumption(self) -> float:
        """
        Estimate current power consumption in Watts.
        
        Uses CPU utilization to estimate power draw based on
        Raspberry Pi 4 power characteristics.
        
        Returns:
            Estimated power consumption in Watts
        """
        cpu_usage = self.get_cpu_usage()
        
        # Linear interpolation between idle and full load
        if cpu_usage <= 50:
            power = self.POWER_IDLE + (self.POWER_CPU_50 - self.POWER_IDLE) * (cpu_usage / 50.0)
        else:
            power = self.POWER_CPU_50 + (self.POWER_CPU_100 - self.POWER_CPU_50) * ((cpu_usage - 50) / 50.0)
        
        # Add GPU/Camera power if camera is likely active
        # (simplified heuristic)
        power += self.POWER_GPU * 0.5
        
        return power
    
    def calculate_energy(self, duration_seconds: float, avg_power: float) -> float:
        """
        Calculate energy consumption in Joules.
        
        Args:
            duration_seconds: Time duration in seconds
            avg_power: Average power consumption in Watts
        
        Returns:
            Energy consumption in Joules
        """
        return avg_power * duration_seconds
    
    def start_monitoring(self):
        """Start continuous hardware monitoring."""
        self.monitoring = True
        self.start_time = time.time()
        self.measurements = []
        logger.info("Hardware monitoring started")
    
    def stop_monitoring(self):
        """Stop continuous hardware monitoring."""
        self.monitoring = False
        if self.start_time:
            duration = time.time() - self.start_time
            logger.info("Hardware monitoring stopped after %.2f seconds", duration)
    
    def take_measurement(self) -> Dict:
        """
        Take a single hardware measurement.
        
        Returns:
            Dictionary with all hardware metrics
        """
        timestamp = time.time()
        cpu_temp = self.get_cpu_temperature()
        cpu_usage = self.get_cpu_usage()
        memory = self.get_memory_usage()
        power = self.estimate_power_consumption()
        
        measurement = {
            'timestamp': timestamp,
            'datetime': datetime.now().isoformat(),
            'cpu_temp_c': cpu_temp,
            'cpu_percent': cpu_usage,
            'ram_used_mb': memory['used_mb'],
            'ram_total_mb': memory['total_mb'],
            'ram_percent': memory['percent'],
            'power_watts': power
        }
        
        if self.monitoring:
            self.measurements.append(measurement)
        
        return measurement
    
    def get_average_metrics(self) -> Dict:
        """
        Calculate average metrics from all measurements.
        
        Returns:
            Dictionary with average values for all metrics
        """
        if not self.measurements:
            logger.warning("No measurements available")
            return {}
        
        num_measurements = len(self.measurements)
        
        avg_metrics = {
            'avg_cpu_temp': sum(m['cpu_temp_c'] for m in self.measurements) / num_measurements,
            'avg_cpu_percent': sum(m['cpu_percent'] for m in self.measurements) / num_measurements,
            'avg_ram_mb': sum(m['ram_used_mb'] for m in self.measurements) / num_measurements,
            'avg_power_watts': sum(m['power_watts'] for m in self.measurements) / num_measurements,
            'num_measurements': num_measurements
        }
        
        # Calculate total energy
        if num_measurements >= 2:
            duration = self.measurements[-1]['timestamp'] - self.measurements[0]['timestamp']
            avg_metrics['total_energy_joules'] = self.calculate_energy(
                duration, avg_metrics['avg_power_watts']
            )
            avg_metrics['duration_seconds'] = duration
        
        logger.debug("Average metrics calculated from %d measurements", num_measurements)
        return avg_metrics
    
    def run_monitor_loop(self, duration_seconds: float = 5) -> List[Dict]:
        """
        Run monitoring loop for specified duration.
        
        Args:
            duration_seconds: How long to monitor
        
        Returns:
            List of all measurements taken
        """
        logger.info("Starting monitor loop for %.1f seconds", duration_seconds)
        self.start_monitoring()
        
        start_time = time.time()
        try:
            while time.time() - start_time < duration_seconds:
                self.take_measurement()
                time.sleep(self.log_interval)
        except KeyboardInterrupt:
            logger.info("Monitor loop interrupted by user")
        finally:
            self.stop_monitoring()
        
        logger.info("Collected %d measurements", len(self.measurements))
        return self.measurements
    
    def get_peak_metrics(self) -> Dict:
        """
        Get peak (maximum) values from all measurements.
        
        Returns:
            Dictionary with peak values
        """
        if not self.measurements:
            return {}
        
        return {
            'peak_cpu_temp': max(m['cpu_temp_c'] for m in self.measurements),
            'peak_cpu_percent': max(m['cpu_percent'] for m in self.measurements),
            'peak_ram_mb': max(m['ram_used_mb'] for m in self.measurements),
            'peak_power_watts': max(m['power_watts'] for m in self.measurements)
        }


def create_monitor(config: Optional[Dict] = None) -> HardwareMonitor:
    """
    Factory function to create HardwareMonitor from config.
    
    Args:
        config: Configuration dictionary (optional)
    
    Returns:
        Configured HardwareMonitor instance
    """
    if config is None:
        return HardwareMonitor()
    
    log_interval = config.get('monitor_interval', 0.1)
    log_file = config.get('log_file', None)
    
    return HardwareMonitor(log_interval=log_interval, log_file=log_file)


if __name__ == "__main__":
    # Test the hardware monitor
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s %(message)s')
    
    monitor = HardwareMonitor(log_interval=0.5)
    
    print("=" * 50)
    print("Hardware Monitor Test")
    print("=" * 50)
    
    # Run monitoring for 5 seconds
    measurements = monitor.run_monitor_loop(duration_seconds=5)
    
    # Display results
    avg_metrics = monitor.get_average_metrics()
    peak_metrics = monitor.get_peak_metrics()
    
    print("\nAverage Metrics:")
    for key, value in avg_metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    
    print("\nPeak Metrics:")
    for key, value in peak_metrics.items():
        print(f"  {key}: {value:.2f}")
    
    print(f"\nTotal measurements: {len(measurements)}")
    print("=" * 50)
