"""
Cloudflare R2 utilities for ScanPiper.
Handles signed URL generation for secure file uploads and downloads.
"""
import os
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from typing import Optional, Union
import io

class R2Utils:
    def __init__(self, bucket_name: str, account_id: str, access_key_id: str, secret_access_key: str, public_endpoint: Optional[str] = None):
        """
        Initialize R2 client with credentials.
        
        Args:
            bucket_name: Name of the R2 bucket
            account_id: Cloudflare Account ID
            access_key_id: R2 Access Key ID
            secret_access_key: R2 Secret Access Key
            public_endpoint: Public URL endpoint for the bucket (e.g., https://pub-xxx.r2.dev or custom domain)
        """
        self.bucket_name = bucket_name
        self.public_endpoint = public_endpoint.rstrip('/') if public_endpoint else None
        
        # Configure the client
        self.s3_client = boto3.client(
            's3',
            endpoint_url=f'https://{account_id}.r2.cloudflarestorage.com',
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
            config=Config(signature_version='s3v4'),
            region_name='auto' # R2 requires region to be 'auto'
        )

    def upload_file_to_r2(self, file_name: str, file_data: Union[bytes, io.BytesIO], content_type: str = None) -> str:
        """
        Upload a file to R2.
        
        Args:
            file_name: The path/name of the file in the bucket
            file_data: The file content as bytes or BytesIO
            content_type: MIME type of the file
            
        Returns:
            Public URL if public_endpoint is set, otherwise the s3 uri
        """
        try:
            # reset pointer if it's a BytesIO object
            if hasattr(file_data, 'seek'):
                file_data.seek(0)
                
            extra_args = {}
            if content_type:
                extra_args['ContentType'] = content_type
                extra_args['CacheControl'] = 'public, max-age=3600' # Similar to GCS implementation

            self.s3_client.upload_fileobj(
                Fileobj=io.BytesIO(file_data) if isinstance(file_data, bytes) else file_data,
                Bucket=self.bucket_name,
                Key=file_name,
                ExtraArgs=extra_args
            )
            
            if self.public_endpoint:
                return f"{self.public_endpoint}/{file_name}"
            else:
                return f"s3://{self.bucket_name}/{file_name}"
                
        except ClientError as e:
            print(f"Error uploading to R2: {e}")
            raise e

    def upload_file_from_filename_to_r2(self, file_name: str, file_path: str, content_type: str = None) -> str:
        """Upload a file from a local path to R2."""
        try:
            extra_args = {}
            if content_type:
                extra_args['ContentType'] = content_type
            
            self.s3_client.upload_file(file_path, self.bucket_name, file_name, ExtraArgs=extra_args)
            
            if self.public_endpoint:
                return f"{self.public_endpoint}/{file_name}"
            else:
                return f"s3://{self.bucket_name}/{file_name}"
        except ClientError as e:
            print(f"Error uploading file to R2: {e}")
            raise e

    def download_file_from_r2(self, r2_url: str) -> bytes:
        """
        Download a file from R2.
        Accepts full URL or path.
        """
        try:
            # Extract key from URL if passed
            key = r2_url
            if self.public_endpoint and r2_url.startswith(self.public_endpoint):
                key = r2_url.replace(f"{self.public_endpoint}/", "")
            elif r2_url.startswith(f"s3://{self.bucket_name}/"):
                key = r2_url.replace(f"s3://{self.bucket_name}/", "")
                
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)
            return response['Body'].read()
        except ClientError as e:
            print(f"Error downloading from R2: {e}")
            raise e

    def generate_signed_url(self, file_name: str, method: str = 'GET', expiration: int = 3600, content_type: str = None) -> str:
        """
        Generate a presigned URL for the S3 object.
        """
        try:
            params = {
                'Bucket': self.bucket_name,
                'Key': file_name
            }
            if method in ['PUT', 'POST'] and content_type:
                params['ContentType'] = content_type

            client_method = 'get_object' if method == 'GET' else 'put_object'
            
            response = self.s3_client.generate_presigned_url(
                client_method,
                Params=params,
                ExpiresIn=expiration
            )
            return response
        except ClientError as e:
            print(f"Error generating signed URL: {e}")
            raise e
            
    def delete_blob(self, file_name: str) -> bool:
        """Delete an object from R2."""
        try:
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=file_name)
            return True
        except ClientError as e:
            print(f"Error deleting object from R2: {e}")
            return False
