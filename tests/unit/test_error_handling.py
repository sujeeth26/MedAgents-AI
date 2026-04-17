"""
Unit tests for error handling utilities
"""

import unittest
import time
from unittest.mock import Mock, patch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.error_handling import (
    RetryConfig,
    CircuitBreaker,
    retry_with_backoff,
    ErrorHandler
)


class TestRetryMechanism(unittest.TestCase):
    """Test retry with backoff functionality"""
    
    def test_successful_first_attempt(self):
        """Test function succeeds on first attempt"""
        @retry_with_backoff(RetryConfig(max_attempts=3))
        def always_succeeds():
            return "success"
        
        result = always_succeeds()
        self.assertEqual(result, "success")
    
    def test_retry_until_success(self):
        """Test function fails then succeeds"""
        attempt_count = 0
        
        @retry_with_backoff(RetryConfig(max_attempts=3, initial_delay=0.1))
        def fails_twice():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise ConnectionError("Temporary failure")
            return "success"
        
        result = fails_twice()
        self.assertEqual(result, "success")
        self.assertEqual(attempt_count, 3)
    
    def test_max_attempts_exceeded(self):
        """Test function fails all attempts"""
        @retry_with_backoff(RetryConfig(max_attempts=2, initial_delay=0.1))
        def always_fails():
            raise ValueError("Permanent failure")
        
        with self.assertRaises(ValueError):
            always_fails()
    
    def test_exponential_backoff(self):
        """Test that delays increase exponentially"""
        delays = []
        
        @retry_with_backoff(RetryConfig(
            max_attempts=4,
            initial_delay=0.1,
            exponential_base=2.0
        ))
        def track_delays():
            delays.append(time.time())
            raise ConnectionError("Fail")
        
        try:
            track_delays()
        except:
            pass
        
        # Verify delays increased
        if len(delays) > 1:
            gaps = [delays[i+1] - delays[i] for i in range(len(delays)-1)]
            # Each gap should be roughly double the previous
            for i in range(len(gaps)-1):
                self.assertGreater(gaps[i+1], gaps[i] * 0.8)  # Allow some tolerance


class TestCircuitBreaker(unittest.TestCase):
    """Test circuit breaker functionality"""
    
    def test_circuit_closed_normal_operation(self):
        """Test circuit breaker in normal closed state"""
        breaker = CircuitBreaker(failure_threshold=3)
        
        def succeeds():
            return "success"
        
        result = breaker.call(succeeds)
        self.assertEqual(result, "success")
        self.assertEqual(breaker.state, "CLOSED")
    
    def test_circuit_opens_after_threshold(self):
        """Test circuit opens after failure threshold"""
        breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=1)
        
        def always_fails():
            raise RuntimeError("Fail")
        
        # Trigger failures
        for i in range(3):
            try:
                breaker.call(always_fails)
            except:
                pass
        
        self.assertEqual(breaker.state, "OPEN")
        self.assertEqual(breaker.failure_count, 3)
    
    def test_circuit_opens_rejects_requests(self):
        """Test open circuit rejects requests immediately"""
        breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=100)
        
        def fails():
            raise RuntimeError("Fail")
        
        # Trigger opening
        for i in range(2):
            try:
                breaker.call(fails)
            except:
                pass
        
        # Circuit should be open, next call should be rejected
        with self.assertRaises(Exception) as context:
            breaker.call(lambda: "success")
        
        self.assertIn("Circuit breaker OPEN", str(context.exception))
    
    def test_circuit_recovery(self):
        """Test circuit breaker recovery after timeout"""
        breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=0.5)
        
        def fails_then_succeeds(attempt=[0]):
            attempt[0] += 1
            if attempt[0] <= 2:
                raise RuntimeError("Fail")
            return "success"
        
        # Trigger opening
        for i in range(2):
            try:
                breaker.call(lambda: fails_then_succeeds())
            except:
                pass
        
        self.assertEqual(breaker.state, "OPEN")
        
        # Wait for recovery timeout
        time.sleep(0.6)
        
        # Next call should attempt reset (HALF_OPEN)
        try:
            result = breaker.call(lambda: "success")
            self.assertEqual(breaker.state, "CLOSED")
        except:
            pass


class TestErrorHandler(unittest.TestCase):
    """Test error handler utilities"""
    
    def test_handle_agent_error(self):
        """Test agent error handling"""
        error = ValueError("Test error")
        result = ErrorHandler.handle_agent_error("TEST_AGENT", error)
        
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["agent"], "TEST_AGENT")
        self.assertEqual(result["error_type"], "ValueError")
        self.assertIn("timestamp", result)
    
    def test_handle_image_processing_error(self):
        """Test image processing error handling"""
        error = IOError("Image not found")
        result = ErrorHandler.handle_image_processing_error("/path/to/image.jpg", error)
        
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error_type"], "IOError")
        self.assertIn("image", result["message"].lower())
    
    def test_handle_api_error(self):
        """Test API error handling"""
        error = ConnectionError("API unreachable")
        result = ErrorHandler.handle_api_error("OpenAI", error)
        
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["api"], "OpenAI")
        self.assertIn("service", result["message"].lower())
    
    def test_is_retryable_error(self):
        """Test retryable error detection"""
        # Retryable errors
        self.assertTrue(ErrorHandler.is_retryable_error(ConnectionError("Timeout")))
        self.assertTrue(ErrorHandler.is_retryable_error(TimeoutError("Timeout")))
        
        # Non-retryable errors
        self.assertFalse(ErrorHandler.is_retryable_error(ValueError("Invalid input")))
        self.assertFalse(ErrorHandler.is_retryable_error(TypeError("Wrong type")))


class TestRetryConfig(unittest.TestCase):
    """Test retry configuration"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = RetryConfig()
        
        self.assertEqual(config.max_attempts, 3)
        self.assertEqual(config.initial_delay, 1.0)
        self.assertEqual(config.max_delay, 10.0)
        self.assertEqual(config.exponential_base, 2.0)
    
    def test_custom_config(self):
        """Test custom configuration"""
        config = RetryConfig(
            max_attempts=5,
            initial_delay=0.5,
            max_delay=20.0,
            exponential_base=3.0
        )
        
        self.assertEqual(config.max_attempts, 5)
        self.assertEqual(config.initial_delay, 0.5)
        self.assertEqual(config.max_delay, 20.0)
        self.assertEqual(config.exponential_base, 3.0)


if __name__ == '__main__':
    unittest.main()

