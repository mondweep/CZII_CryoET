import os
from dotenv import load_dotenv
import boto3
from datetime import datetime, timedelta

# Load environment variables
load_dotenv()

def check_free_tier_usage():
    try:
        # Create session using environment variables
        session = boto3.Session(
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name=os.getenv('AWS_DEFAULT_REGION')
        )

        # Create CloudWatch client to check usage metrics
        cloudwatch = session.client('cloudwatch')
        
        # Create Cost Explorer client
        ce = session.client('ce')

        print("Checking Free Tier Usage...")

        try:
            # Get cost and usage data
            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            end_date = datetime.now().strftime('%Y-%m-%d')
            
            response = ce.get_cost_and_usage(
                TimePeriod={
                    'Start': start_date,
                    'End': end_date
                },
                Granularity='MONTHLY',
                Metrics=['UnblendedCost'],
                GroupBy=[
                    {'Type': 'DIMENSION', 'Key': 'SERVICE'}
                ]
            )
            
            print("\nCost by Service (Last 30 days):")
            for result in response['ResultsByTime']:
                for group in result['Groups']:
                    service = group['Keys'][0]
                    cost = float(group['Metrics']['UnblendedCost']['Amount'])
                    if cost > 0:
                        print(f"{service}: ${cost:.2f}")

        except Exception as e:
            print(f"Unable to get cost data: {str(e)}")
            print("You might not have Cost Explorer permissions.")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_free_tier_usage() 