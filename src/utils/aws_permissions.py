import os
from dotenv import load_dotenv
import boto3
import json

# Load environment variables
load_dotenv()

def check_aws_permissions():
    try:
        # Create session using environment variables
        session = boto3.Session(
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name=os.getenv('AWS_DEFAULT_REGION')
        )

        # Create IAM client
        iam = session.client('iam')
        sts = session.client('sts')

        # Get identity information
        identity = sts.get_caller_identity()
        print(f"Account ID: {identity['Account']}")
        print(f"User ARN: {identity['Arn']}")
        print(f"User ID: {identity['UserId']}")
        
        # Get user name from ARN
        user_name = identity['Arn'].split('/')[-1]
        
        # Get attached policies
        print("\nAttached Policies:")
        policies = iam.list_attached_user_policies(UserName=user_name)
        for policy in policies['AttachedPolicies']:
            policy_details = iam.get_policy(PolicyArn=policy['PolicyArn'])
            policy_version = iam.get_policy_version(
                PolicyArn=policy['PolicyArn'],
                VersionId=policy_details['Policy']['DefaultVersionId']
            )
            print(f"\nPolicy Name: {policy['PolicyName']}")
            print("Permissions:")
            print(json.dumps(policy_version['PolicyVersion']['Document'], indent=2))

    except Exception as e:
        print(f"Error: {e}")
        print("\nNote: If you see an access denied error, your IAM user might not have permission to view policies.")
        print("Try using the following minimal version instead:")
        check_minimal_permissions()

def check_minimal_permissions():
    """Fallback function that checks permissions by attempting various AWS actions"""
    try:
        session = boto3.Session(
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name=os.getenv('AWS_DEFAULT_REGION')
        )

        # Test common permissions
        services_to_test = {
            's3': ['list_buckets', 'get_bucket_location'],
            'iam': ['get_user', 'list_attached_user_policies'],
            'sts': ['get_caller_identity'],
            'ec2': ['describe_instances', 'describe_regions'],
        }

        print("\nTesting common permissions:")
        for service, actions in services_to_test.items():
            client = session.client(service)
            print(f"\n{service.upper()} Permissions:")
            for action in actions:
                try:
                    getattr(client, action)()
                    print(f"✓ {action}: Allowed")
                except Exception as e:
                    print(f"✗ {action}: Denied ({str(e)})")

    except Exception as e:
        print(f"Error in minimal permissions check: {e}")

if __name__ == "__main__":
    check_aws_permissions() 