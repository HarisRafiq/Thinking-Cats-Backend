"""
Google Cloud Storage utilities for ScanPiper.
Handles signed URL generation for secure file uploads and downloads.
"""
import os
from datetime import datetime, timedelta
from typing import Optional
from google.cloud import storage
from google.oauth2 import service_account

# Set key credentials file path
credentials_path = os.path.join(os.path.dirname(__file__), 'gen-lang-client-0331750798-43ac0c8ca808.json')
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path

class GCSUtils:
    def __init__(self, bucket_name: str):
        """Initialize GCS client with service account credentials and set bucket name."""
        self.bucket_name = bucket_name
        credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
        if credentials_path and os.path.exists(credentials_path):
            self.client = storage.Client.from_service_account_json(credentials_path)
        else:
            self.client = None
            print(f"Warning: Could not initialize GCS client: credentials not found.")

    def _get_bucket(self):
        if not self.client:
            raise Exception("GCS client not initialized")
        return self.client.bucket(self.bucket_name)

    def cors_configuration(self):
        """Set a bucket's CORS policies configuration."""
        bucket = self._get_bucket()
        
        # Reload to get latest metadata
        bucket.reload()
        
        # Set CORS configuration
        bucket.cors = [
            {
                "origin": ["*"],
                "responseHeader": [
                    "Content-Type",
                    "Access-Control-Allow-Origin",
                    "x-goog-resumable"
                ],
                "method": ['GET', 'PUT', 'POST', 'HEAD', 'OPTIONS'],
                "maxAgeSeconds": 3600
            }
        ]
        
        # Update the bucket
        bucket.update()
        
        # Reload to verify
        bucket.reload()
        
        print(f"Set CORS policies for bucket {bucket.name} is {bucket.cors}")
        return bucket

    def upload_file_to_gcs(self, file_name: str, file_data, content_type: str = None) -> str:
        """Upload a file to Google Cloud Storage with CORS-friendly metadata."""
        bucket = self._get_bucket()
        blob = bucket.blob(file_name)
        
        # Set content type based on file extension if not provided
        if content_type is None:
            if file_name.endswith('.png'):
                content_type = 'image/png'
            elif file_name.endswith('.jpg') or file_name.endswith('.jpeg'):
                content_type = 'image/jpeg'
            elif file_name.endswith('.gif'):
                content_type = 'image/gif'
            else:
                content_type = 'application/octet-stream'
        
        # Set CORS-friendly metadata
        blob.metadata = {'Cache-Control': 'public, max-age=3600'}
        blob.cache_control = 'public, max-age=3600'
        
        blob.upload_from_file(file_data, content_type=content_type)
        return f"gs://{self.bucket_name}/{file_name}"

    def upload_file_from_filename_to_gcs(self, file_name: str, file_path: str) -> str:
        """Upload a file from a local path to Google Cloud Storage."""
        bucket = self._get_bucket()
        blob = bucket.blob(file_name)
        blob.upload_from_filename(file_path)
        return f"gs://{self.bucket_name}/{file_name}"

    def download_file_from_gcs(self, gcs_url: str) -> bytes:
        """Download a file from Google Cloud Storage."""
        bucket_name, blob_name = gcs_url.replace("gs://", "").split("/", 1)
        if bucket_name != self.bucket_name:
            raise ValueError(f"Bucket name mismatch: {bucket_name} != {self.bucket_name}")
        bucket = self._get_bucket()
        blob = bucket.blob(blob_name)
        return blob.download_as_bytes()
    
    def download_blob_from_gcs(self, blob_name: str) -> bytes:
        """Download a blob from Google Cloud Storage."""
        bucket = self._get_bucket()
        blob = bucket.blob(blob_name)
        return blob.download_as_bytes()

    def upload_content_to_gcs(self, content: str, filename: str) -> str:
        """Upload content to Google Cloud Storage."""
        bucket = self._get_bucket()
        blob = bucket.blob(filename)
        blob.upload_from_string(content)
        return f"gs://{self.bucket_name}/{filename}"

    def list_blobs(self, prefix: str = None, max_results: int = 100) -> list:
        """
        List blobs in a bucket with optional prefix filter.
        """
        if not self.client:
            return []
        try:
            bucket = self._get_bucket()
            blobs = bucket.list_blobs(prefix=prefix, max_results=max_results)
            return [blob.name for blob in blobs]
        except Exception as e:
            print(f"Error listing blobs: {e}")
            return []

    def generate_signed_url(self, blob_name: str, method: str = 'GET', expiration: int = 3600, content_type: str = None) -> str:
        """Generate a signed URL for a blob in Google Cloud Storage."""
        bucket = self._get_bucket()
        blob = bucket.blob(blob_name)
        expiration_time = datetime.utcnow() + timedelta(seconds=expiration)
        headers = {}
        if method in ['PUT', 'POST'] and content_type:
            headers['Content-Type'] = content_type
        signed_url = blob.generate_signed_url(
            version="v4",
            expiration=expiration_time,
            method=method,
            headers=headers if headers else None
        )
        return signed_url

    def generate_upload_signed_url(self, blob_name: str, content_type: str, expiration: int = 3600) -> str:
        """
        Generate a signed URL for uploading a file to GCS.
        """
        bucket = self._get_bucket()
        blob = bucket.blob(blob_name)
        url = blob.generate_signed_url(
            version="v4",
            expiration=datetime.utcnow() + timedelta(seconds=expiration),
            method="PUT",
            content_type=content_type,
        )
        return url
    
    def generate_download_signed_url_for_gcs(self, gcs_url: str, expiration: int = 3600) -> str:
        """
        Generate a signed URL for downloading a file from GCS using its URI.
        """
        bucket_name, blob_name = gcs_url.replace("gs://", "").split("/", 1)
        if bucket_name != self.bucket_name:
            raise ValueError(f"Bucket name mismatch: {bucket_name} != {self.bucket_name}")
        bucket = self._get_bucket()
        blob = bucket.blob(blob_name)
        url = blob.generate_signed_url(
            version="v4",
            expiration=datetime.utcnow() + timedelta(seconds=expiration),
            method="GET"
        )
        return url

    def generate_download_signed_url(self, blob_name: str, expiration: int = 3600) -> str:
        """
        Generate a signed URL for downloading a file from GCS.
        """
        bucket = self._get_bucket()
        blob = bucket.blob(blob_name)
        url = blob.generate_signed_url(
            version="v4",
            expiration=datetime.utcnow() + timedelta(seconds=expiration),
            method="GET"
        )
        return url

    def delete_blob(self, blob_name: str) -> bool:
        """
        Delete a blob from the bucket.
        """
        if not self.client:
            return False
        try:
            bucket = self._get_bucket()
            blob = bucket.blob(blob_name)
            blob.delete()
            return True
        except Exception as e:
            print(f"Error deleting blob {blob_name}: {e}")
            return False

    @staticmethod
    def load_credentials_from_file():
        """Loads Google Cloud credentials from a service account JSON file."""
        try:
            return service_account.Credentials.from_service_account_file(credentials_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Service account file not found: {credentials_path}")
        except Exception as e:
            raise RuntimeError(f"Error loading credentials from {credentials_path}: {e}")
