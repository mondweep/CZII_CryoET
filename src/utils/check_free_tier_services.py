import os
from dotenv import load_dotenv
import boto3
from datetime import datetime, timedelta

def check_free_tier_services():
    try:
        session = boto3.Session(
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name=os.getenv('AWS_DEFAULT_REGION')
        )

        # Create Pricing client
        pricing = session.client('pricing', region_name='us-east-1')  # Pricing API is only available in us-east-1
        
        # Create Free Tier client
        freetier = session.client('freetier')

        try:
            # Get Free Tier eligibility and usage
            response = freetier.get_free_tier_usage(
                TimePeriod={
                    'Start': (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'),
                    'End': datetime.now().strftime('%Y-%m-%d')
                }
            )
            
            print("\nFree Tier Usage:")
            for service in response.get('FreeUsageByService', []):
                print(f"\nService: {service['ServiceName']}")
                print(f"Usage Type: {service['UsageType']}")
                print(f"Free Tier Limit: {service['FreeUsageLimit']}")
                print(f"Current Usage: {service['CurrentUsage']}")

        except Exception as e:
            print(f"\nUnable to get Free Tier usage directly: {str(e)}")
            print("\nTrying to get service information from Pricing API...")
            
            # Get services with free tier offerings
            response = pricing.describe_services()
            print("\nAWS Services with potential Free Tier offerings:")
            for service in response['Services']:
                try:
                    filters = [
                        {'Type': 'TERM_MATCH', 'Field': 'usagetype', 'Value': 'free'}
                    ]
                    
                    pricing.get_products(
                        ServiceCode=service['ServiceCode'],
                        Filters=filters,
                        MaxResults=1
                    )
                    print(f"- {service['ServiceCode']}")
                except:
                    continue

    except Exception as e:
        print(f"Error: {e}")
        print("\nNote: To view Free Tier information, you might need to:")
        print("1. Enable Cost Explorer in AWS Console")
        print("2. Wait 24 hours for data to become available")
        print("3. Visit AWS Free Tier page for manual checking: https://aws.amazon.com/free/")

if __name__ == "__main__":
    load_dotenv()
    check_free_tier_services() 