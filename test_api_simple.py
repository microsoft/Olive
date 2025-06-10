#!/usr/bin/env python
"""Simple test to validate the Olive Python API structure."""

def test_api_structure():
    """Test basic API structure without dependencies."""
    print("Testing API structure...")
    
    try:
        from olive.api.workflow import (
            auto_opt,
            capture_onnx,
            finetune,
            generate_adapter,
            quantize,
            run,
            session_params_tuning,
        )
        print("✓ API functions import successfully")
        
        # Test function signatures
        import inspect
        
        funcs = [auto_opt, capture_onnx, finetune, generate_adapter, quantize, run, session_params_tuning]
        for func in funcs:
            sig = inspect.signature(func)
            assert len(sig.parameters) > 0, f"{func.__name__} should have parameters"
            print(f"✓ {func.__name__} has proper signature")
        
        print("✓ All API tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ API test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_api_structure()
    if not success:
        exit(1)