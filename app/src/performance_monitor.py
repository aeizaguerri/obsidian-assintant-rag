"""
Performance monitoring and metrics for the RAG system.

This module provides tools to monitor and analyze the performance of
different components of the RAG system.
"""

import time
import functools
from typing import Dict, List, Callable, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque

@dataclass
class PerformanceMetric:
    """Performance metric data"""
    name: str
    duration: float
    timestamp: float
    metadata: Dict = field(default_factory=dict)

class PerformanceMonitor:
    """Monitor and track performance metrics"""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history))
        self.active_timers: Dict[str, float] = {}
    
    def start_timer(self, operation: str) -> float:
        """Start timing an operation"""
        start_time = time.time()
        self.active_timers[operation] = start_time
        return start_time
    
    def end_timer(self, operation: str, metadata: Dict = None) -> float:
        """End timing an operation and record metric"""
        end_time = time.time()
        start_time = self.active_timers.pop(operation, end_time)
        duration = end_time - start_time
        
        metric = PerformanceMetric(
            name=operation,
            duration=duration,
            timestamp=end_time,
            metadata=metadata or {}
        )
        
        self.metrics[operation].append(metric)
        return duration
    
    def record_metric(self, operation: str, duration: float, metadata: Dict = None):
        """Manually record a metric"""
        metric = PerformanceMetric(
            name=operation,
            duration=duration,
            timestamp=time.time(),
            metadata=metadata or {}
        )
        self.metrics[operation].append(metric)
    
    def get_stats(self, operation: str) -> Dict:
        """Get statistics for an operation"""
        if operation not in self.metrics or not self.metrics[operation]:
            return {}
        
        durations = [m.duration for m in self.metrics[operation]]
        
        return {
            'count': len(durations),
            'avg_duration': sum(durations) / len(durations),
            'min_duration': min(durations),
            'max_duration': max(durations),
            'total_duration': sum(durations),
            'recent_avg': sum(durations[-10:]) / min(10, len(durations))
        }
    
    def get_all_stats(self) -> Dict[str, Dict]:
        """Get statistics for all operations"""
        return {op: self.get_stats(op) for op in self.metrics.keys()}
    
    def clear_metrics(self, operation: str = None):
        """Clear metrics for operation or all operations"""
        if operation:
            self.metrics[operation].clear()
        else:
            self.metrics.clear()

def timed(operation_name: str = None, monitor: PerformanceMonitor = None):
    """Decorator to time function execution"""
    def decorator(func: Callable) -> Callable:
        nonlocal operation_name
        if operation_name is None:
            operation_name = f"{func.__module__}.{func.__name__}"
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            perf_monitor = monitor or get_global_monitor()
            
            start_time = perf_monitor.start_timer(operation_name)
            try:
                result = func(*args, **kwargs)
                perf_monitor.end_timer(operation_name, {'success': True})
                return result
            except Exception as e:
                perf_monitor.end_timer(operation_name, {'success': False, 'error': str(e)})
                raise
        
        return wrapper
    return decorator

# Global performance monitor
_global_monitor = None

def get_global_monitor() -> PerformanceMonitor:
    """Get or create global performance monitor"""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = PerformanceMonitor()
    return _global_monitor

def get_performance_report() -> str:
    """Generate a human-readable performance report"""
    monitor = get_global_monitor()
    stats = monitor.get_all_stats()
    
    if not stats:
        return "No performance data available."
    
    report_lines = ["# RAG System Performance Report\n"]
    
    for operation, data in stats.items():
        report_lines.append(f"## {operation}")
        report_lines.append(f"- **Total calls:** {data['count']}")
        report_lines.append(f"- **Average duration:** {data['avg_duration']:.3f}s")
        report_lines.append(f"- **Min duration:** {data['min_duration']:.3f}s")
        report_lines.append(f"- **Max duration:** {data['max_duration']:.3f}s")
        report_lines.append(f"- **Recent average:** {data['recent_avg']:.3f}s")
        report_lines.append("")
    
    return "\n".join(report_lines)