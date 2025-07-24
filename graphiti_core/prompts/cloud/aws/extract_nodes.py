"""
AWS-specific prompts for node extraction
"""

import json
from typing import Any, Protocol, TypedDict
from pydantic import BaseModel, Field
from ...models import Message, PromptFunction, PromptVersion


class ExtractedEntity(BaseModel):
    name: str = Field(..., description='Name of the extracted entity')
    entity_type_id: int = Field(
        description='ID of the classified entity type. '
        'Must be one of the provided entity_type_id integers.',
    )


class ExtractedEntities(BaseModel):
    extracted_entities: list[ExtractedEntity] = Field(..., description='List of extracted entities')


class MissedEntities(BaseModel):
    missed_entities: list[str] = Field(..., description="Names of entities that weren't extracted")


class EntityClassificationTriple(BaseModel):
    uuid: str = Field(description='UUID of the entity')
    name: str = Field(description='Name of the entity')
    entity_type: str | None = Field(
        default=None, description='Type of the entity. Must be one of the provided types or None'
    )


class EntityClassification(BaseModel):
    entity_classifications: list[EntityClassificationTriple] = Field(
        ..., description='List of entities classification triples.'
    )


class Prompt(Protocol):
    extract_message: PromptVersion
    extract_json: PromptVersion
    extract_text: PromptVersion
    reflexion: PromptVersion
    classify_nodes: PromptVersion
    extract_attributes: PromptVersion


class Versions(TypedDict):
    extract_message: PromptFunction
    extract_json: PromptFunction
    extract_text: PromptFunction
    reflexion: PromptFunction
    classify_nodes: PromptFunction
    extract_attributes: PromptFunction


def extract_message(context: dict[str, Any]) -> list[Message]:
    sys_prompt = """You are an AI assistant specialized in extracting AWS infrastructure entities.
    Your primary focus is identifying AWS resources, their configurations, and relationships from AWS configurations and IaC files."""

    user_prompt = f"""
<PREVIOUS MESSAGES>
{json.dumps([ep for ep in context['previous_episodes']], indent=2)}
</PREVIOUS MESSAGES>

<CURRENT MESSAGE>
{context['episode_content']}
</CURRENT MESSAGE>

<ENTITY TYPES>
{context['entity_types']}
</ENTITY TYPES>

You are analyzing AWS infrastructure resources. Extract entities focusing on:

## PRIMARY EXTRACTION TARGETS:

1. **EC2 & Compute Resources**:
   - EC2 instances (with instance IDs like "i-1234567890abcdef0")
   - Auto Scaling Groups (ASG names)
   - ECS clusters and services
   - EKS clusters and node groups
   - Lambda functions
   - Batch compute environments

2. **VPC & Networking Resources**:
   - VPCs (with VPC IDs like "vpc-12345678")
   - Subnets (with subnet IDs like "subnet-12345678")
   - Security Groups (with SG IDs like "sg-12345678")
   - Network ACLs
   - Route Tables
   - Internet Gateways and NAT Gateways
   - VPC Endpoints

3. **Load Balancing & Traffic Management**:
   - Application Load Balancers (ALB)
   - Network Load Balancers (NLB)
   - Target Groups
   - API Gateway APIs
   - CloudFront distributions

4. **Storage & Database Resources**:
   - S3 buckets
   - EBS volumes (with volume IDs like "vol-12345678")
   - RDS instances and clusters
   - DynamoDB tables
   - ElastiCache clusters
   - EFS file systems
   - Glacier vaults

5. **Security & Identity**:
   - IAM roles and policies
   - IAM users and groups
   - KMS keys and aliases
   - Cognito User Pools
   - Secrets Manager secrets
   - Certificate Manager certificates

6. **Messaging & Integration**:
   - SQS queues
   - SNS topics and subscriptions
   - EventBridge rules
   - Step Functions state machines
   - SQS dead letter queues

7. **Monitoring & Logging**:
   - CloudWatch log groups
   - CloudWatch dashboards
   - CloudWatch alarms
   - X-Ray traces
   - Config rules

## AWS-SPECIFIC RULES:
- Always include resource IDs when available (e.g., "i-1234567890abcdef0", "vpc-12345678")
- Use resource names as primary identifiers when IDs aren't available
- Include region information in entity names when relevant (e.g., "prod-web-alb-us-east-1")
- Extract IAM roles and policies as entities
- Capture resource tags that indicate ownership or purpose
- Include environment prefixes (prod-, staging-, dev-)

## AWS NAMING CONVENTIONS:
- EC2 instances: use instance names or IDs (e.g., "prod-web-server", "i-0a1b2c3d")
- VPC resources: use VPC names or IDs (e.g., "prod-vpc", "vpc-12345678")
- S3 buckets: use bucket names (e.g., "my-app-storage", "prod-logs-bucket")
- RDS instances: use instance names (e.g., "prod-database", "staging-postgres")

## DO NOT EXTRACT:
- IP addresses or CIDR blocks as separate entities
- Temporary resources or build artifacts
- Configuration values or parameters
- Cost information or billing details
- Generic AWS service names without specific instances

{context['custom_prompt']}
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt),
    ]


def extract_json(context: dict[str, Any]) -> list[Message]:
    sys_prompt = """You are an AI assistant specialized in extracting AWS infrastructure entities from JSON configurations.
    Focus on CloudFormation templates, Terraform state files, AWS CLI output, and AWS API responses."""

    user_prompt = f"""
<SOURCE DESCRIPTION>:
{context['source_description']}
</SOURCE DESCRIPTION>
<JSON>
{context['episode_content']}
</JSON>
<ENTITY TYPES>
{context['entity_types']}
</ENTITY TYPES>

You are analyzing JSON from AWS infrastructure. This could be:
- CloudFormation templates
- Terraform state files (.tfstate)
- AWS CLI output (aws ec2 describe-instances, etc.)
- AWS API responses
- CDK output
- AWS Config snapshots

## EXTRACTION FOCUS FOR AWS JSON:

1. **From CloudFormation Templates**:
   - Logical resource names from "Resources" section
   - Resource types (AWS::EC2::Instance, AWS::S3::Bucket, etc.)
   - Stack names and nested stacks
   - Output values representing important resources

2. **From Terraform State**:
   - Resource names from "resources" array
   - Instance IDs from "instances" objects
   - Resource types and their identifiers
   - Module names representing infrastructure components

3. **From AWS CLI Output**:
   - Instance IDs from describe-instances
   - VPC IDs from describe-vpcs
   - Security Group IDs from describe-security-groups
   - S3 bucket names from list-buckets
   - RDS instance names from describe-db-instances

4. **From AWS API Responses**:
   - Resource identifiers and names
   - Service-specific resource types
   - Tagged resources with meaningful names

## AWS JSON SPECIFIC RULES:
- Extract resource names from appropriate fields (name, id, identifier, arn)
- Include resource type in extraction when it helps identify the entity
- For nested resources, maintain the hierarchy in naming
- Skip parameter definitions and variable declarations
- Focus on actual provisioned resources, not templates
- Extract from both "Resources" and "Outputs" sections in CloudFormation

{context['custom_prompt']}
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt),
    ]


def extract_text(context: dict[str, Any]) -> list[Message]:
    sys_prompt = """You are an AI assistant specialized in extracting AWS infrastructure entities from documentation and configuration files.
    Focus on AWS infrastructure documentation, runbooks, and deployment guides."""

    user_prompt = f"""
<TEXT>
{context['episode_content']}
</TEXT>
<ENTITY TYPES>
{context['entity_types']}
</ENTITY TYPES>

You are analyzing text about AWS infrastructure. This could be:
- AWS infrastructure documentation
- Deployment guides and runbooks
- Architecture diagrams descriptions
- AWS Well-Architected reviews
- Cloud migration plans
- AWS service documentation

## EXTRACTION FOCUS FOR AWS TEXT:

1. **Infrastructure Components**:
   - VPC names and network segments
   - EC2 instance groups or Auto Scaling Groups
   - RDS cluster names and instances
   - S3 bucket names
   - Load balancer names (ALB, NLB)

2. **AWS Managed Services**:
   - EKS cluster names
   - ECS cluster and service names
   - Lambda function names
   - API Gateway names
   - CloudFront distribution names
   - SQS queue names

3. **Architecture Entities**:
   - Environment names (prod, staging, dev)
   - Application tier names (web, app, data)
   - Availability zones or regions
   - Disaster recovery sites
   - AWS account names or IDs

## AWS TEXT SPECIFIC RULES:
- Extract specific resource names, not generic descriptions
- Include environment prefixes/suffixes (prod-, -staging)
- Capture multi-region deployments as separate entities
- Focus on currently deployed resources, not planned ones
- Extract from architecture diagrams if textually described
- Include AWS service names when they represent specific instances

## AWS NAMING STANDARDS:
- Use full resource names including environment (e.g., "prod-web-asg")
- Include region in name if multi-region (e.g., "cache-cluster-us-east-1")
- Maintain AWS naming conventions (lowercase, hyphens)
- Use canonical names for well-known AWS services
- Include account context when relevant (e.g., "prod-account-vpc")

{context['custom_prompt']}
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt),
    ]


def reflexion(context: dict[str, Any]) -> list[Message]:
    sys_prompt = """You are an AI assistant that determines which AWS entities have not been extracted from the given context"""

    user_prompt = f"""
<PREVIOUS MESSAGES>
{json.dumps([ep for ep in context['previous_episodes']], indent=2)}
</PREVIOUS MESSAGES>
<CURRENT MESSAGE>
{context['episode_content']}
</CURRENT MESSAGE>

<EXTRACTED ENTITIES>
{context['extracted_entities']}
</EXTRACTED ENTITIES>

Given the above previous messages, current message, and list of extracted entities; determine if any AWS infrastructure entities haven't been extracted.

Focus on missing:
- EC2 instances and compute resources
- VPC and networking components
- Storage and database resources
- Load balancers and traffic management
- Security and IAM resources
- Messaging and integration services
- Monitoring and logging resources
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt),
    ]


def classify_nodes(context: dict[str, Any]) -> list[Message]:
    sys_prompt = """You are an AI assistant that classifies AWS entity nodes given the context from which they were extracted"""

    user_prompt = f"""
    <PREVIOUS MESSAGES>
    {json.dumps([ep for ep in context['previous_episodes']], indent=2)}
    </PREVIOUS MESSAGES>
    <CURRENT MESSAGE>
    {context['episode_content']}
    </CURRENT MESSAGE>
    
    <EXTRACTED ENTITIES>
    {context['extracted_entities']}
    </EXTRACTED ENTITIES>
    
    <ENTITY TYPES>
    {context['entity_types']}
    </ENTITY TYPES>
    
    Given the above conversation, extracted entities, and provided entity types and their descriptions, classify the extracted AWS entities.
    
    Guidelines:
    1. Each entity must have exactly one type
    2. Only use the provided ENTITY TYPES as types, do not use additional types to classify entities.
    3. If none of the provided entity types accurately classify an extracted node, the type should be set to None
    4. Consider AWS-specific context when classifying (e.g., EC2 instances, S3 buckets, RDS instances)
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt),
    ]


def extract_attributes(context: dict[str, Any]) -> list[Message]:
    return [
        Message(
            role='system',
            content='You are a helpful assistant that extracts AWS entity properties from the provided text.',
        ),
        Message(
            role='user',
            content=f"""

        <MESSAGES>
        {json.dumps(context['previous_episodes'], indent=2)}
        {json.dumps(context['episode_content'], indent=2)}
        </MESSAGES>

        Given the above MESSAGES and the following ENTITY, update any of its attributes based on the information provided
        in MESSAGES. Use the provided attribute descriptions to better understand how each attribute should be determined.

        Guidelines:
        1. Do not hallucinate entity property values if they cannot be found in the current context.
        2. Only use the provided MESSAGES and ENTITY to set attribute values.
        3. The summary attribute represents a summary of the ENTITY, and should be updated with new information about the Entity from the MESSAGES. 
            Summaries must be no longer than 250 words.
        4. Consider AWS-specific attributes like instance types, VPC IDs, security group configurations, etc.
        
        <ENTITY>
        {context['node']}
        </ENTITY>
        """,
        ),
    ]


versions: Versions = {
    'extract_message': extract_message,
    'extract_json': extract_json,
    'extract_text': extract_text,
    'reflexion': reflexion,
    'classify_nodes': classify_nodes,
    'extract_attributes': extract_attributes,
} 