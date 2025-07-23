"""
Interpolation strategies for MSI data processing.

This module contains different interpolation strategies that can be used
for resampling mass spectrometry data to optimal mass axes.
"""

from .base_strategy import InterpolationStrategy, QualityMetrics
from .pchip_strategy import PchipInterpolationStrategy
from .adaptive_strategy import AdaptiveInterpolationStrategy

# Strategy registry for easy access
AVAILABLE_STRATEGIES = {
    'pchip': PchipInterpolationStrategy,
    'adaptive': AdaptiveInterpolationStrategy
}

def create_interpolation_strategy(strategy_name: str, **kwargs) -> InterpolationStrategy:
    """
    Factory function to create interpolation strategies.
    
    Args:
        strategy_name: Name of the strategy ('pchip', 'adaptive')
        **kwargs: Additional arguments passed to strategy constructor
        
    Returns:
        Interpolation strategy instance
        
    Raises:
        ValueError: If strategy name is not recognized
    """
    if strategy_name not in AVAILABLE_STRATEGIES:
        available = list(AVAILABLE_STRATEGIES.keys())
        raise ValueError(f"Unknown strategy '{strategy_name}'. Available: {available}")
    
    strategy_class = AVAILABLE_STRATEGIES[strategy_name]
    return strategy_class(**kwargs)

def get_strategy_info() -> dict:
    """
    Get information about all available strategies.
    
    Returns:
        Dictionary with strategy information
    """
    info = {}
    for name, strategy_class in AVAILABLE_STRATEGIES.items():
        # Create temporary instance to get info
        temp_instance = strategy_class(validate_quality=False)
        info[name] = temp_instance.get_performance_info()
    return info

__all__ = [
    'InterpolationStrategy',
    'QualityMetrics', 
    'PchipInterpolationStrategy',
    'AdaptiveInterpolationStrategy',
    'create_interpolation_strategy',
    'get_strategy_info',
    'AVAILABLE_STRATEGIES'
]