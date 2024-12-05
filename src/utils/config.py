class Config:
    """Configuration class to store the configuration from a YAML file."""

    def __init__(self, config_dict):
        """Initialize configuration."""
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)

    def __getattr__(self, key):
        """Get attribute."""
        try:
            return super().__getattr__(key)
        except AttributeError:
            return None

    def __str__(self):
        """String representation."""
        return str(self.__dict__)

    def __repr__(self):
        """String representation."""
        return self.__str__()

    def to_dict(self):
        """Convert to dictionary."""
        output = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                output[key] = value.to_dict()
            else:
                output[key] = value
        return output 