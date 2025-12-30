#!/usr/bin/env python3
"""
Script to configure CORS on the GCS bucket for social image generation.
This enables canvas-based operations like GIF generation from the frontend.
"""
import sys
import os
from modular_agent.utils.gcs_utils import GCSUtils

def main():
    bucket_name = os.getenv("GCS_BUCKET_NAME", "thinking-cats-images")
    print(f"Configuring CORS for bucket: {bucket_name}")
    
    try:
        gcs_utils = GCSUtils(bucket_name)
        bucket = gcs_utils.cors_configuration()
        
        print("\n✅ CORS configuration successful!")
        print(f"Bucket: {bucket.name}")
        print(f"CORS policies: {bucket.cors}")
        
        return 0
    except Exception as e:
        print(f"\n❌ Error configuring CORS: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
