from model_core.llm_client import LLMClient
import sys

def test_connection():
    print("Testing LLM Connection...")
    client = LLMClient()
    
    # Simple direct test
    msg = [{"role": "user", "content": "Return the word 'ConnectivityVerified'."}]
    response = client._query_llm(msg)
    print(f"Raw Response: {response}")
    
    if "ConnectivityVerified" in response:
        print("SUCCESS: Connection verified.")
    else:
        print("WARNING: Unexpected response content.")

def test_generation():
    print("\nTesting Factor Generation...")
    client = LLMClient()
    factors = client.generate_initial_hypothesis(n=2)
    print(f"Generated Factors: {factors}")
    
    if factors and isinstance(factors, list) and len(factors) > 0:
         print("SUCCESS: Parsed factors correctly.")
    else:
         print("FAILURE: Could not parse factors.")

if __name__ == "__main__":
    test_connection()
    test_generation()
