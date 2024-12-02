# AWS Tools

A collection of Python tools for AWS account management and monitoring.

## Setup

1. Clone the repository: 
bash
git clone [your-repo-url]
cd aws-tools

2. Create a virtual environment:
```bash
python -m venv aws-tools-env
source aws-tools-env/bin/activate  On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file with your AWS credentials:
```bash
AWS_ACCESS_KEY_ID=your_access_key_id
AWS_SECRET_ACCESS_KEY=your_secret_access_key
AWS_DEFAULT_REGION=your_region
```

## Features

- AWS Permissions checking
- Free Tier usage monitoring
- Cost Explorer integration

## Usage

Example usage:
```bash
python check_free_tier_services.py
```