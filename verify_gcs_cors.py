#!/usr/bin/env python3
"""
Verify CORS configuration on the GCS bucket.
"""
import sys
import os
from modular_agent.utils.gcs_utils import GCSUtils

def main():
    bucket_name = os.getenv("GCS_BUCKET_NAME", "thinking-cats-images")
    print(f"Verifying CORS for bucket: {bucket_name}\n")
    
    try:
        gcs_utils = GCSUtils(bucket_name)
        bucket = gcs_utils._get_bucket()
        
        print(f"✅ Bucket found: {bucket.name}")
        print(f"\nCurrent CORS configuration:")
        if bucket.cors:
            for i, cors_rule in enumerate(bucket.cors, 1):
                print(f"\n  Rule {i}:")
                print(f"    Origins: {cors_rule.get('origin', [])}")
                print(f"    Methods: {cors_rule.get('method', [])}")
                print(f"    Response Headers: {cors_rule.get('responseHeader', [])}")
                print(f"    Max Age: {cors_rule.get('maxAgeSeconds', 'N/A')} seconds")
        else:
            print("  ⚠️  No CORS rules configured!")
            
        print("\n" + "="*60)
        print("To test CORS manually, run:")
        print(f"  curl -H 'Origin: http://localhost:5173' \\")
        print(f"       -H 'Access-Control-Request-Method: GET' \\")
        print(f"       -X OPTIONS \\")
        print(f"       -I 'https://storage.googleapis.com/{bucket_name}/'")
        print("="*60)
        
        return 0
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
