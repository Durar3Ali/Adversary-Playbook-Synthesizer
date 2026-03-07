"""
Diagnostic script to test Gemini API connection and list available models.
Run this to diagnose API key and model availability issues.
"""

import os
import sys

# Load .env so GEMINI_API_KEY is available when running this script directly.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv is optional here; key can be set in the shell instead

try:
    import google.generativeai as genai
except ImportError:
    print("[ERROR] google-generativeai not installed. Install with: pip install google-generativeai")
    sys.exit(1)

def test_gemini_api():
    """Test Gemini API connection and list available models"""
    
    # Get API key from environment
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("[ERROR] GEMINI_API_KEY environment variable not set.")
        print("Please set it in your .env file or environment variables.")
        return False
    
    print(f"[OK] API Key found: {api_key[:10]}...{api_key[-4:]}")
    print("\n" + "="*60)
    print("Testing Gemini API Connection...")
    print("="*60 + "\n")
    
    try:
        # Configure API
        genai.configure(api_key=api_key)
        print("[OK] API configured successfully\n")
    except Exception as e:
        print(f"[ERROR] Failed to configure API: {e}\n")
        return False
    
    # Try to list available models
    print("Listing available models...")
    print("-" * 60)
    try:
        models = list(genai.list_models())
        if models:
            print(f"[OK] Found {len(models)} available model(s):\n")
            available_generative = []
            for model in models:
                model_name = model.name if hasattr(model, 'name') else str(model)
                methods = getattr(model, 'supported_generation_methods', [])
                if 'generateContent' in methods:
                    available_generative.append(model_name)
                    print(f"  [OK] {model_name}")
                    print(f"    Methods: {', '.join(methods)}")
                else:
                    print(f"  [-] {model_name} (no generateContent support)")
            
            print(f"\n[OK] Found {len(available_generative)} model(s) with generateContent support:")
            for model_name in available_generative:
                print(f"    - {model_name}")
        else:
            print("[ERROR] No models found. This might indicate an API key issue.")
            return False
    except Exception as e:
        print(f"[ERROR] Failed to list models: {e}\n")
        print("This might mean:")
        print("  1. API key doesn't have permission to list models")
        print("  2. API key is invalid")
        print("  3. Network/connection issue")
        return False
    
    # Test model initialization
    print("\n" + "="*60)
    print("Testing Model Initialization...")
    print("="*60 + "\n")
    
    models_to_test = [
        "gemini-1.5-flash",
        "gemini-1.5-pro",
        "gemini-1.5-flash-latest",
        "gemini-1.5-pro-latest",
        "gemini-2.0-flash-exp",
        "gemini-pro",
    ]
    
    working_models = []
    for model_name in models_to_test:
        try:
            model = genai.GenerativeModel(model_name)
            print(f"[OK] {model_name}: OK")
            working_models.append(model_name)
            
            # Try a simple test generation
            try:
                response = model.generate_content("Say 'test'")
                if response and response.text:
                    print(f"  -> Test generation successful")
            except Exception as e:
                print(f"  -> Model created but generation failed: {e}")
        except Exception as e:
            print(f"[FAIL] {model_name}: FAILED - {str(e)[:100]}")
    
    print("\n" + "="*60)
    if working_models:
        print(f"[SUCCESS] Found {len(working_models)} working model(s):")
        for model in working_models:
            print(f"    - {model}")
        print(f"\nRecommendation: Use '{working_models[0]}' in your code.")
        return True
    else:
        print("[FAILED] No working models found.")
        print("\nTroubleshooting:")
        print("1. Check your API key in Google Cloud Console")
        print("2. Ensure Gemini API is enabled for your project")
        print("3. Verify API key has correct permissions")
        print("4. Try generating a new API key")
        return False

if __name__ == "__main__":
    print("\n" + "="*60)
    print("Gemini API Diagnostic Tool")
    print("="*60 + "\n")
    
    success = test_gemini_api()
    
    print("\n" + "="*60)
    if success:
        print("Diagnostic completed successfully!")
    else:
        print("Diagnostic found issues. Please review the errors above.")
    print("="*60 + "\n")
    
    sys.exit(0 if success else 1)

