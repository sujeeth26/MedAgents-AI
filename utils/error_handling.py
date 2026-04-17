"""
Error handling utilities with retry mechanisms and circuit breakers.
Provides robust error handling for API calls and critical operations.
"""

import time
import logging
from functools import wraps
from typing import Callable, Any, Optional, Type
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class RetryConfig:
    """Configuration for retry behavior"""
    def __init__(
        self,
        max_attempts: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 10.0,
        exponential_base: float = 2.0,
        exceptions: tuple = (Exception,)
    ):
        self.max_attempts = max_attempts
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.exceptions = exceptions


class CircuitBreaker:
    """
    Circuit breaker pattern implementation to prevent cascading failures.
    
    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Too many failures, requests are immediately rejected
    - HALF_OPEN: Testing if service has recovered
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: Type[Exception] = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        if self.state == "OPEN":
            if self._should_attempt_reset():
                self.state = "HALF_OPEN"
            else:
                raise Exception(f"Circuit breaker OPEN for {func.__name__}")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset"""
        if self.last_failure_time is None:
            return True
        
        time_since_failure = (datetime.now() - self.last_failure_time).total_seconds()
        return time_since_failure >= self.recovery_timeout
    
    def _on_success(self):
        """Handle successful call"""
        self.failure_count = 0
        self.state = "CLOSED"
    
    def _on_failure(self):
        """Handle failed call"""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            logger.warning(
                f"Circuit breaker opened after {self.failure_count} failures"
            )


def retry_with_backoff(config: Optional[RetryConfig] = None):
    """
    Decorator to retry a function with exponential backoff.
    
    Usage:
        @retry_with_backoff(RetryConfig(max_attempts=3))
        def my_api_call():
            # ... code that might fail
    """
    if config is None:
        config = RetryConfig()
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            delay = config.initial_delay
            
            for attempt in range(1, config.max_attempts + 1):
                try:
                    result = func(*args, **kwargs)
                    if attempt > 1:
                        logger.info(
                            f"{func.__name__} succeeded on attempt {attempt}"
                        )
                    return result
                    
                except config.exceptions as e:
                    last_exception = e
                    
                    if attempt == config.max_attempts:
                        logger.error(
                            f"{func.__name__} failed after {attempt} attempts: {e}"
                        )
                        break
                    
                    logger.warning(
                        f"{func.__name__} failed (attempt {attempt}/{config.max_attempts}): {e}. "
                        f"Retrying in {delay}s..."
                    )
                    
                    time.sleep(delay)
                    delay = min(delay * config.exponential_base, config.max_delay)
            
            raise last_exception
        
        return wrapper
    return decorator


class ErrorHandler:
    """
    Centralized error handling for consistent error responses.
    """
    
    @staticmethod
    def handle_agent_error(agent_name: str, error: Exception) -> dict:
        """Handle errors from agent execution"""
        logger.error(f"Agent {agent_name} error: {str(error)}", exc_info=True)
        
        return {
            "status": "error",
            "agent": agent_name,
            "error_type": type(error).__name__,
            "message": "An error occurred processing your request. Please try again.",
            "timestamp": datetime.now().isoformat()
        }
    
    @staticmethod
    def handle_image_processing_error(image_path: str, error: Exception) -> dict:
        """Handle errors from image processing"""
        logger.error(f"Image processing error for {image_path}: {str(error)}", exc_info=True)
        
        return {
            "status": "error",
            "error_type": type(error).__name__,
            "message": "Failed to process the uploaded image. Please ensure it's a valid medical image and try again.",
            "timestamp": datetime.now().isoformat()
        }
    
    @staticmethod
    def handle_api_error(api_name: str, error: Exception) -> dict:
        """Handle errors from external API calls"""
        logger.error(f"API {api_name} error: {str(error)}", exc_info=True)
        
        return {
            "status": "error",
            "api": api_name,
            "error_type": type(error).__name__,
            "message": "External service temporarily unavailable. Please try again later.",
            "timestamp": datetime.now().isoformat()
        }
    
    @staticmethod
    def is_retryable_error(error: Exception) -> bool:
        """Determine if an error is retryable"""
        retryable_errors = [
            "ConnectionError",
            "Timeout",
            "ServiceUnavailable",
            "RateLimitError",
            "APIError"
        ]
        
        error_name = type(error).__name__
        return any(retryable in error_name for retryable in retryable_errors)


# Pre-configured retry decorators for common scenarios
retry_api_call = retry_with_backoff(
    RetryConfig(
        max_attempts=3,
        initial_delay=1.0,
        max_delay=5.0,
        exceptions=(ConnectionError, TimeoutError)
    )
)

retry_image_processing = retry_with_backoff(
    RetryConfig(
        max_attempts=2,
        initial_delay=0.5,
        max_delay=2.0,
        exceptions=(IOError, OSError, RuntimeError)
    )
)

retry_db_operation = retry_with_backoff(
    RetryConfig(
        max_attempts=3,
        initial_delay=0.5,
        max_delay=3.0,
        exceptions=(ConnectionError, TimeoutError)
    )
)


# Example usage:
if __name__ == "__main__":
    # Test retry decorator
    @retry_api_call
    def flaky_api_call():
        import random
        if random.random() < 0.7:
            raise ConnectionError("API temporarily unavailable")
        return "Success!"
    
    try:
        result = flaky_api_call()
        print(f"Result: {result}")
    except Exception as e:
        print(f"Failed: {e}")
    
    # Test circuit breaker
    breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=5)
    
    def unreliable_service():
        import random
        if random.random() < 0.8:
            raise Exception("Service error")
        return "Success!"
    
    for i in range(10):
        try:
            result = breaker.call(unreliable_service)
            print(f"Call {i+1}: {result}")
        except Exception as e:
            print(f"Call {i+1}: {e}")
        time.sleep(1)

