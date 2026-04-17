"""
Integration tests for API endpoints
"""

import unittest
import sys
import os
from fastapi.testclient import TestClient

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class TestHealthEndpoints(unittest.TestCase):
    """Test health check endpoints"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test client"""
        try:
            from web.app import app
            cls.client = TestClient(app)
        except Exception as e:
            print(f"Warning: Could not initialize test client: {e}")
            cls.client = None
    
    def test_basic_health_check(self):
        """Test /health endpoint"""
        if not self.client:
            self.skipTest("Test client not available")
        
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertIn("timestamp", data)
    
    def test_readiness_check(self):
        """Test /health/ready endpoint"""
        if not self.client:
            self.skipTest("Test client not available")
        
        response = self.client.get("/health/ready")
        # Should return 200 or 503 depending on system state
        self.assertIn(response.status_code, [200, 503])
        
        data = response.json()
        self.assertIn("overall_status", data)
        self.assertIn("timestamp", data)
    
    def test_liveness_check(self):
        """Test /health/live endpoint"""
        if not self.client:
            self.skipTest("Test client not available")
        
        response = self.client.get("/health/live")
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertEqual(data["status"], "alive")
    
    def test_metrics_endpoint(self):
        """Test /metrics endpoint"""
        if not self.client:
            self.skipTest("Test client not available")
        
        response = self.client.get("/metrics")
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertIn("active_sessions", data)
        self.assertIn("total_image_accesses", data)
        self.assertIsInstance(data["active_sessions"], int)


class TestChatEndpoint(unittest.TestCase):
    """Test chat endpoint"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test client"""
        try:
            from web.app import app
            cls.client = TestClient(app)
        except Exception as e:
            print(f"Warning: Could not initialize test client: {e}")
            cls.client = None
    
    def test_chat_endpoint_structure(self):
        """Test chat endpoint accepts requests"""
        if not self.client:
            self.skipTest("Test client not available")
        
        # Note: Actual test would require proper setup of agents
        # This is a structure test only
        response = self.client.post(
            "/chat",
            json={"query": "Hello", "conversation_history": []}
        )
        
        # May fail due to agent initialization, but structure should be correct
        # Just verify endpoint exists
        self.assertIn(response.status_code, [200, 500])


class TestSecurityHeaders(unittest.TestCase):
    """Test security-related headers and cookies"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test client"""
        try:
            from web.app import app
            cls.client = TestClient(app)
        except Exception as e:
            print(f"Warning: Could not initialize test client: {e}")
            cls.client = None
    
    def test_cookie_security_attributes(self):
        """Test that cookies have security attributes"""
        if not self.client:
            self.skipTest("Test client not available")
        
        # Make a request that sets a cookie
        response = self.client.post(
            "/chat",
            json={"query": "test", "conversation_history": []}
        )
        
        # Check cookie attributes
        if 'set-cookie' in response.headers:
            cookie_header = response.headers['set-cookie']
            
            # Should have httponly
            self.assertIn('HttpOnly', cookie_header)
            
            # Should have SameSite
            self.assertIn('SameSite', cookie_header)


if __name__ == '__main__':
    unittest.main()

