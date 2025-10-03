from app.generative.engine import GenAIServices
from botocore.exceptions import ClientError

S3_SESSION  = GenAIServices.chatBedrock(return_session=True)
S3_CLIENT   = S3_SESSION.client('s3')

class S3Uploader:
    
    @staticmethod
    def upload_file(file_path, bucket_name, object_name):
        """Uploads a file from a local path to S3."""
        try:
            S3_CLIENT.upload_file(file_path, bucket_name, object_name)
            return True
        except FileNotFoundError:
            print(f"Error: The file {file_path} was not found.")
            return False
        except ClientError as e:
            print(f"AWS Client Error uploading file: {e}")
            return False

    @staticmethod
    def download_file(bucket_name, object_name, file_path):
        """Downloads a file from S3 to a local path."""
        try:
            S3_CLIENT.download_file(bucket_name, object_name, file_path)
            return True
        except ClientError as e:
            print(f"AWS Client Error downloading file: {e}")
            return False
            
    @staticmethod
    def put_object(file_content, bucket_name, object_name):
        """Uploads in-memory content (bytes) to S3."""
        try:
            S3_CLIENT.put_object(Body=file_content, Bucket=bucket_name, Key=object_name)
            return True
        except ClientError as e:
            print(f"AWS Client Error putting object: {e}")
            return False
            
    @staticmethod
    def get_object(bucket_name, object_name):
        """Gets an object's content (bytes) from S3."""
        try:
            response = S3_CLIENT.get_object(Bucket=bucket_name, Key=object_name)
            return response['Body'].read()
        except ClientError as e:
            print(f"AWS Client Error getting object: {e}")
            return None

    