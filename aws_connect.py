import os
from dotenv import load_dotenv
import boto3

# Load environment variables
load_dotenv()

def connect_to_aws():
    try:
        # Create session using environment variables
        session = boto3.Session(
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name=os.getenv('AWS_DEFAULT_REGION')
        )

        # Create clients
        s3 = session.client('s3')
        sts = session.client('sts')

        # Test connection
        identity = sts.get_caller_identity()
        print(f"Connected as: {identity['Arn']}")

        # List buckets
        response = s3.list_buckets()
        print("\nS3 Buckets:")
        for bucket in response['Buckets']:
            print(bucket['Name'])

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    connect_to_aws() 