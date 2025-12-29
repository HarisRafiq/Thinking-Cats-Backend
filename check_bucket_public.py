#!/usr/bin/env python3
"""
Make bucket objects publicly accessible (bypasses CORS).
"""
import sys
import os
from google.cloud import storage

def main():
    bucket_name = os.getenv("GCS_BUCKET_NAME", "thinking-cats-images")
    print(f"Checking public access for bucket: {bucket_name}\n")
    
    try:
        # Initialize client
        credentials_path = os.path.join(os.path.dirname(__file__), 'utils', 'gen-lang-client-0331750798-43ac0c8ca808.json')
        client = storage.Client.from_service_account_json(credentials_path)
        bucket = client.bucket(bucket_name)
        
        # Check current IAM policy
        policy = bucket.get_iam_policy(requested_policy_version=3)
        
        print("Current IAM bindings:")
        for binding in policy.bindings:
            print(f"  Role: {binding['role']}")
            members = list(binding.get('members', []))
            print(f"  Members: {members[:3] if len(members) > 3 else members}...")  # Show first 3
            
        # Check if allUsers has objectViewer role
        has_public_read = any(
            'allUsers' in binding.get('members', []) and 
            binding['role'] == 'roles/storage.objectViewer'
            for binding in policy.bindings
        )
        
        if has_public_read:
            print("\n✅ Bucket objects are publicly readable!")
            print("   This means CORS should not be an issue for public URLs.")
        else:
            print("\n⚠️  Bucket objects are NOT publicly readable.")
            print("   You may need to make them public or configure CORS properly.")
            print("\n   To make objects public, run:")
            print(f"   gcloud storage buckets add-iam-policy-binding gs://{bucket_name} \\")
            print(f"       --member=allUsers --role=roles/storage.objectViewer")
        
        return 0
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
