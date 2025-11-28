from dataclasses import dataclass


@dataclass
class RetryConfig:
    """
    Configuration for retry and exponential backoff during batch inference.

    Attributes:
        max_retries: Number of retry attempts AFTER the first failed attempt.
                     Total attempts = 1 (initial) + max_retries.
        initial_backoff: Initial sleep duration in seconds before retrying.
        backoff_multiplier: Factor applied to the backoff after each failure.
        max_backoff: Maximum amount of time to sleep between retries.
        TODO: Set config attributes to env vars so multiple engines can be spun up with differing configs
    """
    max_retries: int = 3
    initial_backoff: float = 1.0
    backoff_multiplier: float = 2.0
    max_backoff: float = 30.0


@dataclass
class InferenceConfig:
    model_name: str
    task: str
    batch_size: int = 8
    max_new_tokens: int = 32
    device: str = "cuda"
    input_column: str = "text"
    retry: RetryConfig = RetryConfig()
