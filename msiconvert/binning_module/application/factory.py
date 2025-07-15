# msiconvert/binning_module/application/factory.py
"""Factory for creating strategy instances."""

from typing import Dict, Type

from ..domain.strategies import BinningStrategy, LinearTOFStrategy, ReflectorTOFStrategy
from ..exceptions import UnknownStrategyError


class StrategyFactory:
    """Factory for creating binning strategy instances based on model type."""
    
    # Registry of available strategies
    _strategies: Dict[str, Type[BinningStrategy]] = {
        'linear': LinearTOFStrategy,
        'reflector': ReflectorTOFStrategy,
    }
    
    @classmethod
    def create_strategy(cls, model_type: str) -> BinningStrategy:
        """
        Create a strategy instance based on model type string.
        
        Parameters
        ----------
        model_type : str
            Instrument type identifier
            
        Returns
        -------
        BinningStrategy
            Concrete strategy instance
            
        Raises
        ------
        UnknownStrategyError
            If model_type is not recognized
        """
        # Normalize to lowercase
        model_type = model_type.lower()
        
        if model_type not in cls._strategies:
            available = ', '.join(cls._strategies.keys())
            raise UnknownStrategyError(
                f"Unknown model type '{model_type}'. "
                f"Available types: {available}"
            )
        
        # Create and return instance
        strategy_class = cls._strategies[model_type]
        return strategy_class()
    
    @classmethod
    def register_strategy(cls, model_type: str, strategy_class: Type[BinningStrategy]):
        """
        Register a new strategy type (for extensibility).
        
        Parameters
        ----------
        model_type : str
            Identifier for the strategy
        strategy_class : Type[BinningStrategy]
            Strategy class to register
        """
        cls._strategies[model_type.lower()] = strategy_class
    
    @classmethod
    def available_strategies(cls) -> list[str]:
        """
        Get list of available strategy types.
        
        Returns
        -------
        list[str]
            List of registered model types
        """
        return list(cls._strategies.keys())