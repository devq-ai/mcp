#!/usr/bin/env python3
"""
Simplified Logfire Integration Test for Task 2.1

This test validates core Logfire SDK integration without complex dependencies:
- Logfire credentials loading
- Basic instrumentation setup
- FastAPI integration
- Performance verification
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import logfire
from fastapi import FastAPI
from fastapi.testclient import TestClient


def load_logfire_credentials() -> Dict[str, str]:
    """Load Logfire credentials from credentials file with fallback to environment variables."""
    credentials_path = Path(".logfire/logfire_credentials.json")
    
    try:
        if credentials_path.exists():
            with open(credentials_path, 'r') as f:
                credentials = json.load(f)
                return {
                    'token': credentials.get('write-token'),
                    'project_name': credentials.get('project_name', 'agentical'),
                    'project_url': credentials.get('project_url'),
                    'api_url': credentials.get('logfire_api_url')
                }
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Warning: Failed to load Logfire credentials from file: {e}")
    
    # Fallback to environment variables
    return {
        'token': os.getenv("LOGFIRE_TOKEN"),
        'project_name': os.getenv("LOGFIRE_PROJECT_NAME", "agentical"),
        'project_url': None,
        'api_url': None
    }


def test_credentials_loading():
    """Test 1: Verify Logfire credentials loading"""
    print("Test 1: Logfire Credentials Loading")
    
    try:
        # Test credentials file exists
        credentials_path = Path(".logfire/logfire_credentials.json")
        if not credentials_path.exists():
            print("  ❌ FAIL: Logfire credentials file not found")
            return False
        
        # Test credentials loading function
        credentials = load_logfire_credentials()
        
        # Verify required fields
        if not credentials.get('token'):
            print("  ❌ FAIL: No token found in credentials")
            return False
        
        # Verify token format
        token = credentials['token']
        if not token.startswith('pylf_v1_us_'):
            print(f"  ❌ FAIL: Invalid token format: {token[:20]}...")
            return False
        
        print(f"  ✅ PASS: Successfully loaded credentials for project: {credentials['project_name']}")
        return True
        
    except Exception as e:
        print(f"  ❌ FAIL: Exception during credentials loading: {str(e)}")
        return False


def test_logfire_configuration():
    """Test 2: Verify Logfire configuration"""
    print("Test 2: Logfire Configuration")
    
    try:
        # Load credentials
        logfire_creds = load_logfire_credentials()
        
        # Configure Logfire
        logfire.configure(
            token=logfire_creds['token'],
            project_name=logfire_creds['project_name'],
            service_name="agentical-test",
            environment="test"
        )
        
        # Test span creation
        with logfire.span("test_configuration", test_type="integration"):
            logfire.info("Logfire configuration test", status="running")
        
        print("  ✅ PASS: Logfire configuration successful")
        return True
        
    except Exception as e:
        print(f"  ❌ FAIL: Exception during configuration: {str(e)}")
        return False


def test_fastapi_instrumentation():
    """Test 3: Verify FastAPI auto-instrumentation"""
    print("Test 3: FastAPI Auto-Instrumentation")
    
    try:
        # Create test FastAPI app
        app = FastAPI(title="Test App", version="1.0.0")
        
        # Enable Logfire instrumentation
        logfire.instrument_fastapi(app, capture_headers=True)
        
        # Add test endpoint
        @app.get("/test")
        async def test_endpoint():
            with logfire.span("test_endpoint_processing"):
                return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}
        
        @app.get("/health")
        async def health():
            return {"status": "healthy"}
        
        # Test with client
        client = TestClient(app)
        response = client.get("/test")
        
        if response.status_code != 200:
            print(f"  ❌ FAIL: Test endpoint returned {response.status_code}")
            return False
        
        data = response.json()
        if "status" not in data:
            print("  ❌ FAIL: Test endpoint response missing 'status' field")
            return False
        
        print("  ✅ PASS: FastAPI auto-instrumentation working")
        return True
        
    except Exception as e:
        print(f"  ❌ FAIL: Exception during instrumentation test: {str(e)}")
        return False


def test_httpx_instrumentation():
    """Test 4: Verify HTTPx instrumentation"""
    print("Test 4: HTTPx Instrumentation")
    
    try:
        # Enable HTTPx instrumentation
        logfire.instrument_httpx()
        
        print("  ✅ PASS: HTTPx instrumentation enabled without errors")
        return True
        
    except Exception as e:
        print(f"  ❌ FAIL: Exception during HTTPx instrumentation: {str(e)}")
        return False


def test_sqlalchemy_instrumentation():
    """Test 5: Verify SQLAlchemy instrumentation"""
    print("Test 5: SQLAlchemy Instrumentation")
    
    try:
        # Enable SQLAlchemy instrumentation
        logfire.instrument_sqlalchemy()
        
        print("  ✅ PASS: SQLAlchemy instrumentation enabled without errors")
        return True
        
    except Exception as e:
        print(f"  ❌ FAIL: Exception during SQLAlchemy instrumentation: {str(e)}")
        return False


def test_performance_overhead():
    """Test 6: Verify acceptable performance overhead"""
    print("Test 6: Performance Overhead")
    
    try:
        # Create test app with instrumentation
        app = FastAPI()
        logfire.instrument_fastapi(app)
        
        @app.get("/perf")
        async def perf_test():
            return {"test": "performance"}
        
        client = TestClient(app)
        
        # Measure performance
        num_requests = 5
        response_times = []
        
        for _ in range(num_requests):
            start_time = time.time()
            response = client.get("/perf")
            end_time = time.time()
            
            if response.status_code != 200:
                print(f"  ❌ FAIL: Performance test request failed")
                return False
            
            response_time = (end_time - start_time) * 1000  # Convert to ms
            response_times.append(response_time)
        
        avg_time = sum(response_times) / len(response_times)
        max_time = max(response_times)
        
        # Check if performance is acceptable (100ms for development)
        if avg_time > 100:
            print(f"  ❌ FAIL: Average response time too high: {avg_time:.2f}ms")
            return False
        
        print(f"  ✅ PASS: Performance acceptable - Avg: {avg_time:.2f}ms, Max: {max_time:.2f}ms")
        return True
        
    except Exception as e:
        print(f"  ❌ FAIL: Exception during performance test: {str(e)}")
        return False


def test_structured_logging():
    """Test 7: Verify structured logging with context"""
    print("Test 7: Structured Logging Context")
    
    try:
        # Test manual span creation with context
        with logfire.span("structured_logging_test", test_id="task_2_1"):
            logfire.info("Testing structured logging", 
                        component="logfire_integration",
                        test_phase="validation")
            
            # Nested span
            with logfire.span("nested_operation", operation="validation"):
                logfire.debug("Nested operation completed")
        
        print("  ✅ PASS: Structured logging with context working")
        return True
        
    except Exception as e:
        print(f"  ❌ FAIL: Exception during structured logging test: {str(e)}")
        return False


def run_all_tests():
    """Run all Logfire integration tests"""
    print("🚀 Starting Logfire Integration Tests for Task 2.1")
    print("=" * 60)
    
    tests = [
        ("Credentials Loading", test_credentials_loading),
        ("Logfire Configuration", test_logfire_configuration),
        ("FastAPI Instrumentation", test_fastapi_instrumentation),
        ("HTTPx Instrumentation", test_httpx_instrumentation),
        ("SQLAlchemy Instrumentation", test_sqlalchemy_instrumentation),
        ("Performance Overhead", test_performance_overhead),
        ("Structured Logging", test_structured_logging),
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed_tests += 1
        except Exception as e:
            print(f"  ❌ FAIL: Unexpected exception in {test_name}: {str(e)}")
    
    print("\n" + "=" * 60)
    print(f"📊 LOGFIRE INTEGRATION TEST RESULTS")
    print("=" * 60)
    
    success_rate = (passed_tests / total_tests) * 100
    overall_success = passed_tests == total_tests
    
    print(f"Overall Status: {'✅ PASS' if overall_success else '❌ FAIL'}")
    print(f"Tests Passed: {passed_tests}/{total_tests} ({success_rate:.1f}%)")
    
    if overall_success:
        print("\n🎯 RESULT: SUCCESS")
        print("✅ Task 2.1 Logfire SDK Integration is complete and functional!")
        print("✅ Ready to proceed with Task 2.2 (Structured Logging Implementation)")
    else:
        print("\n🎯 RESULT: FAILED")
        print("❌ Some tests failed - review and fix issues before proceeding")
    
    return overall_success


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)