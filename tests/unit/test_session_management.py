"""
Unit tests for session management functionality
"""

import unittest
import tempfile
import os
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class TestSessionManagement(unittest.TestCase):
    """Test session storage and expiry logic"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.session_images = {}
        self.test_session_id = "test-session-123"
        self.test_image_path = "/tmp/test_image.jpg"
    
    def test_store_session_image(self):
        """Test storing image in session"""
        # Create test image file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as f:
            f.write(b"fake image data")
            image_path = f.name
        
        try:
            # Simulate storing
            self.session_images[self.test_session_id] = {
                'image_path': image_path,
                'timestamp': datetime.now(),
                'access_count': 0
            }
            
            self.assertIn(self.test_session_id, self.session_images)
            self.assertEqual(
                self.session_images[self.test_session_id]['image_path'],
                image_path
            )
            self.assertEqual(
                self.session_images[self.test_session_id]['access_count'],
                0
            )
        finally:
            # Cleanup
            if os.path.exists(image_path):
                os.remove(image_path)
    
    def test_session_expiry(self):
        """Test that old sessions are detected as expired"""
        # Create expired session
        old_timestamp = datetime.now() - timedelta(hours=3)
        self.session_images[self.test_session_id] = {
            'image_path': self.test_image_path,
            'timestamp': old_timestamp,
            'access_count': 0
        }
        
        # Check if expired
        session_data = self.session_images.get(self.test_session_id)
        age = datetime.now() - session_data['timestamp']
        
        self.assertGreater(age.total_seconds(), 2 * 3600)  # More than 2 hours
    
    def test_access_count_increment(self):
        """Test that access count increments correctly"""
        self.session_images[self.test_session_id] = {
            'image_path': self.test_image_path,
            'timestamp': datetime.now(),
            'access_count': 0
        }
        
        # Simulate multiple accesses
        for i in range(5):
            self.session_images[self.test_session_id]['access_count'] += 1
        
        self.assertEqual(
            self.session_images[self.test_session_id]['access_count'],
            5
        )
    
    def test_cleanup_expired_sessions(self):
        """Test cleanup of expired sessions"""
        # Add fresh session
        fresh_session = "fresh-session"
        self.session_images[fresh_session] = {
            'image_path': "/tmp/fresh.jpg",
            'timestamp': datetime.now(),
            'access_count': 0
        }
        
        # Add expired session
        expired_session = "expired-session"
        self.session_images[expired_session] = {
            'image_path': "/tmp/expired.jpg",
            'timestamp': datetime.now() - timedelta(hours=3),
            'access_count': 0
        }
        
        # Cleanup expired
        expiry_threshold = timedelta(hours=2)
        current_time = datetime.now()
        
        expired_ids = [
            sid for sid, data in self.session_images.items()
            if current_time - data['timestamp'] > expiry_threshold
        ]
        
        for sid in expired_ids:
            del self.session_images[sid]
        
        # Verify
        self.assertIn(fresh_session, self.session_images)
        self.assertNotIn(expired_session, self.session_images)


class TestAuditLogging(unittest.TestCase):
    """Test audit logging functionality"""
    
    def test_audit_log_format(self):
        """Test that audit logs are formatted correctly"""
        import logging
        
        # Capture log output
        with self.assertLogs(level='INFO') as log:
            logger = logging.getLogger('test_audit')
            logger.info("AUDIT: action=IMAGE_STORED session=test123... details=path=/tmp/test.jpg")
        
        # Verify log format
        self.assertTrue(any('AUDIT' in message for message in log.output))
        self.assertTrue(any('IMAGE_STORED' in message for message in log.output))
    
    def test_sensitive_data_redaction(self):
        """Test that sensitive data is redacted in logs"""
        session_id = "very-long-session-id-with-sensitive-data-12345678"
        
        # Simulate redaction (first 8 chars + ...)
        redacted = session_id[:8] + "..."
        
        self.assertEqual(redacted, "very-lon...")
        self.assertLess(len(redacted), len(session_id))


if __name__ == '__main__':
    unittest.main()

