"""
Cloud resources specific prompts for node extraction
"""

import json
from typing import Any, Protocol, TypedDict
from pydantic import BaseModel, Field
from .models import Message, PromptFunction, PromptVersion


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
    sys_prompt = """You are an AI assistant specialized in extracting cloud infrastructure entities.
    Your primary focus is identifying cloud resources, their configurations, and relationships from cloud provider configurations and IaC files."""

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

You are analyzing cloud infrastructure resources. Extract entities focusing on:

## PRIMARY EXTRACTION TARGETS:

1. **Compute Resources**:
   - EC2 instances (with instance IDs and names)
   - Virtual Machines (Azure VMs, GCE instances)
   - Container instances (ECS, AKS, GKE)
   - Lambda functions / Cloud Functions / Azure Functions
   - Auto Scaling Groups

2. **Networking Resources**:
   - VPCs and Virtual Networks
   - Subnets (with CIDR blocks if mentioned)
   - Security Groups and Network ACLs
   - Load Balancers (ALB, NLB, Application Gateway)
   - API Gateways and endpoints

3. **Storage Resources**:
   - S3 buckets / Azure Blob containers / GCS buckets
   - EBS volumes / Managed Disks
   - RDS instances / Azure SQL / Cloud SQL
   - DynamoDB tables / Cosmos DB / Firestore
   - ElastiCache clusters / Azure Cache

4. **Managed Services**:
   - Message queues (SQS, Service Bus, Pub/Sub)
   - Kubernetes clusters (EKS, AKS, GKE)
   - Container registries (ECR, ACR, GCR)
   - CDN distributions (CloudFront, Azure CDN)

## CLOUD-SPECIFIC RULES:
- Always include resource IDs when available (e.g., "i-1234567890abcdef0")
- Use resource names as primary identifiers
- Include region/zone information in entity names when relevant
- Extract IAM roles and service accounts as entities
- Capture resource tags that indicate ownership or purpose

## NAMING CONVENTIONS:
- AWS: use resource names or IDs (e.g., "prod-web-alb", "i-0a1b2c3d")
- Azure: use resource names (e.g., "prodWebVM", "storageaccount01")
- GCP: use resource names (e.g., "prod-instance-group", "my-gke-cluster")

## DO NOT EXTRACT:
- IP addresses or CIDR blocks as separate entities
- Temporary resources or build artifacts
- Configuration values or parameters
- Cost information or billing details

{context['custom_prompt']}
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt),
    ]


def extract_json(context: dict[str, Any]) -> list[Message]:
    sys_prompt = """You are an AI assistant specialized in extracting cloud infrastructure entities from JSON configurations.
    Focus on Terraform state files, CloudFormation templates, ARM templates, and cloud API responses."""

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

You are analyzing JSON from cloud infrastructure. This could be:
- Terraform state files (.tfstate)
- CloudFormation templates
- ARM templates
- Cloud provider API responses
- Kubernetes manifests
- Infrastructure as Code configurations

## EXTRACTION FOCUS FOR CLOUD JSON:

1. **From Terraform State**:
   - Resource names from "resources" array
   - Instance IDs from "instances" objects
   - Resource types and their identifiers
   - Module names representing infrastructure components

2. **From CloudFormation/ARM**:
   - Logical resource names from "Resources" section
   - Resource types (AWS::EC2::Instance, etc.)
   - Stack names and nested stacks
   - Output values representing important resources

3. **From Kubernetes Manifests**:
   - Deployment names
   - Service names
   - Ingress resources
   - ConfigMaps and Secrets (names only)
   - Persistent Volume Claims

## CLOUD JSON SPECIFIC RULES:
- Extract resource names from appropriate fields (name, id, identifier)
- Include resource type in extraction when it helps identify the entity
- For nested resources, maintain the hierarchy in naming
- Skip parameter definitions and variable declarations
- Focus on actual provisioned resources, not templates

{context['custom_prompt']}
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt),
    ]


def extract_text(context: dict[str, Any]) -> list[Message]:
    sys_prompt = """You are an AI assistant specialized in extracting cloud infrastructure entities from documentation and configuration files.
    Focus on infrastructure documentation, runbooks, and deployment guides."""

    user_prompt = f"""
<TEXT>
{context['episode_content']}
</TEXT>
<ENTITY TYPES>
{context['entity_types']}
</ENTITY TYPES>

You are analyzing text about cloud infrastructure. This could be:
- Infrastructure documentation
- Deployment guides
- Architecture diagrams descriptions
- Runbooks and operation procedures
- Cloud migration plans

## EXTRACTION FOCUS FOR CLOUD TEXT:

1. **Infrastructure Components**:
   - VPC names and network segments
   - Compute instance groups or clusters
   - Database cluster names
   - Storage bucket names
   - Load balancer names

2. **Managed Services**:
   - Kubernetes cluster names (EKS, AKS, GKE)
   - Serverless function names
   - API Gateway names
   - CDN distribution names
   - Message queue names

3. **Architecture Entities**:
   - Environment names (prod, staging, dev)
   - Application tier names (web, app, data)
   - Availability zones or regions
   - Disaster recovery sites

## CLOUD TEXT SPECIFIC RULES:
- Extract specific resource names, not generic descriptions
- Include environment prefixes/suffixes (prod-, -staging)
- Capture multi-region deployments as separate entities
- Focus on currently deployed resources, not planned ones
- Extract from architecture diagrams if textually described

## NAMING STANDARDS:
- Use full resource names including environment (e.g., "prod-web-asg")
- Include region in name if multi-region (e.g., "cache-cluster-us-east-1")
- Maintain cloud provider naming conventions
- Use canonical names for well-known services

{context['custom_prompt']}
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt),
    ]


def reflexion(context: dict[str, Any]) -> list[Message]:
    sys_prompt = """You are an AI assistant that determines which entities have not been extracted from the given context"""

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

Given the above previous messages, current message, and list of extracted entities; determine if any entities haven't been
extracted.
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt),
    ]


def classify_nodes(context: dict[str, Any]) -> list[Message]:
    sys_prompt = """You are an AI assistant that classifies entity nodes given the context from which they were extracted"""

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
    
    Given the above conversation, extracted entities, and provided entity types and their descriptions, classify the extracted entities.
    
    Guidelines:
    1. Each entity must have exactly one type
    2. Only use the provided ENTITY TYPES as types, do not use additional types to classify entities.
    3. If none of the provided entity types accurately classify an extracted node, the type should be set to None
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt),
    ]


def extract_attributes(context: dict[str, Any]) -> list[Message]:
    return [
        Message(
            role='system',
            content='You are a helpful assistant that extracts entity properties from the provided text.',
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