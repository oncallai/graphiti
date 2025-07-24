"""
GCP-specific prompts for node extraction
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
    sys_prompt = """You are an AI assistant specialized in extracting GCP infrastructure entities.
    Your primary focus is identifying GCP resources, their configurations, and relationships from GCP configurations and IaC files."""

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

You are analyzing GCP infrastructure resources. Extract entities focusing on:

## PRIMARY EXTRACTION TARGETS:

1. **Compute Engine & Compute Resources**:
   - Compute Engine instances (VM names)
   - Instance Groups and Instance Templates
   - Container Instances and Container Groups
   - Google Kubernetes Engine (GKE) clusters
   - Cloud Functions
   - Cloud Run services
   - App Engine applications

2. **VPC & Networking Resources**:
   - VPC networks (network names)
   - Subnets (with subnet names)
   - Firewall rules
   - Route tables
   - Load balancers
   - Cloud NAT gateways
   - Cloud VPN gateways
   - Cloud Interconnect

3. **Storage & Database Resources**:
   - Cloud Storage buckets
   - Persistent disks
   - Cloud SQL instances
   - Cloud Spanner instances
   - Firestore databases
   - BigQuery datasets
   - Memorystore instances
   - Cloud Filestore instances

4. **App Engine & Cloud Run Resources**:
   - App Engine applications
   - App Engine services and versions
   - Cloud Run services
   - Cloud Run revisions
   - Cloud Build triggers
   - Container Registry repositories

5. **Security & Identity**:
   - Service accounts
   - IAM roles and policies
   - Cloud KMS keys and key rings
   - Identity Platform projects
   - Secret Manager secrets
   - Certificate Manager certificates

6. **Messaging & Integration**:
   - Pub/Sub topics and subscriptions
   - Cloud Tasks queues
   - Cloud Scheduler jobs
   - Workflows
   - API Gateway APIs

7. **Monitoring & Management**:
   - Cloud Monitoring workspaces
   - Logging buckets and sinks
   - Cloud Trace traces
   - Cloud Debugger snapshots
   - Cloud Profiler profiles

## GCP-SPECIFIC RULES:
- Use resource names as primary identifiers (GCP uses names rather than IDs)
- Include project information when relevant
- Include region/zone information in entity names when relevant (e.g., "prod-web-app-us-central1")
- Extract service accounts and IAM resources as entities
- Capture resource labels that indicate ownership or purpose
- Include environment prefixes (prod-, staging-, dev-)

## GCP NAMING CONVENTIONS:
- Compute instances: use instance names (e.g., "prod-web-server", "staging-app-instance")
- VPC networks: use network names (e.g., "prod-network", "staging-vpc")
- Cloud Storage: use bucket names (e.g., "my-app-storage", "prod-logs-bucket")
- Cloud SQL: use instance names (e.g., "prod-database", "staging-postgres")

## DO NOT EXTRACT:
- IP addresses or CIDR blocks as separate entities
- Temporary resources or build artifacts
- Configuration values or parameters
- Cost information or billing details
- Generic GCP service names without specific instances

{context['custom_prompt']}
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt),
    ]


def extract_json(context: dict[str, Any]) -> list[Message]:
    sys_prompt = """You are an AI assistant specialized in extracting GCP infrastructure entities from JSON configurations.
    Focus on Deployment Manager templates, Terraform state files, gcloud CLI output, and GCP API responses."""

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

You are analyzing JSON from GCP infrastructure. This could be:
- Deployment Manager templates
- Terraform state files (.tfstate)
- gcloud CLI output (gcloud compute instances list, etc.)
- GCP API responses
- Cloud Build configurations
- GCP Resource Manager queries

## EXTRACTION FOCUS FOR GCP JSON:

1. **From Deployment Manager Templates**:
   - Resource names from "resources" array
   - Resource types (compute.v1.instance, storage.v1.bucket, etc.)
   - Project names and configurations
   - Output values representing important resources

2. **From Terraform State**:
   - Resource names from "resources" array
   - Resource types and their identifiers
   - Module names representing infrastructure components
   - GCP-specific resource attributes

3. **From gcloud CLI Output**:
   - Instance names from compute instances list
   - Network names from compute networks list
   - Storage bucket names from storage buckets list
   - App Engine app names from app describe
   - Project names from config list

4. **From GCP API Responses**:
   - Resource identifiers and names
   - Service-specific resource types
   - Labeled resources with meaningful names

## GCP JSON SPECIFIC RULES:
- Extract resource names from appropriate fields (name, id, selfLink)
- Include resource type in extraction when it helps identify the entity
- For nested resources, maintain the hierarchy in naming
- Skip parameter definitions and variable declarations
- Focus on actual provisioned resources, not templates
- Extract from both "resources" and "outputs" sections in Deployment Manager templates

{context['custom_prompt']}
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt),
    ]


def extract_text(context: dict[str, Any]) -> list[Message]:
    sys_prompt = """You are an AI assistant specialized in extracting GCP infrastructure entities from documentation and configuration files.
    Focus on GCP infrastructure documentation, runbooks, and deployment guides."""

    user_prompt = f"""
<TEXT>
{context['episode_content']}
</TEXT>
<ENTITY TYPES>
{context['entity_types']}
</ENTITY TYPES>

You are analyzing text about GCP infrastructure. This could be:
- GCP infrastructure documentation
- Deployment guides and runbooks
- Architecture diagrams descriptions
- GCP Well-Architected reviews
- Cloud migration plans
- GCP service documentation

## EXTRACTION FOCUS FOR GCP TEXT:

1. **Infrastructure Components**:
   - VPC network names and network segments
   - Compute instance groups or managed instance groups
   - Cloud SQL cluster names and instances
   - Cloud Storage bucket names
   - Load balancer names (HTTP(S), TCP/UDP, SSL)

2. **GCP Managed Services**:
   - GKE cluster names
   - App Engine application names
   - Cloud Run service names
   - Cloud Functions names
   - API Gateway names
   - Cloud CDN backend bucket names

3. **Architecture Entities**:
   - Environment names (prod, staging, dev)
   - Application tier names (web, app, data)
   - GCP regions or zones
   - Disaster recovery sites
   - GCP project names or IDs

## GCP TEXT SPECIFIC RULES:
- Extract specific resource names, not generic descriptions
- Include environment prefixes/suffixes (prod-, -staging)
- Capture multi-region deployments as separate entities
- Focus on currently deployed resources, not planned ones
- Extract from architecture diagrams if textually described
- Include GCP service names when they represent specific instances

## GCP NAMING STANDARDS:
- Use full resource names including environment (e.g., "prod-web-instance")
- Include region in name if multi-region (e.g., "cache-cluster-us-central1")
- Maintain GCP naming conventions (lowercase, hyphens)
- Use canonical names for well-known GCP services
- Include project context when relevant (e.g., "prod-project-vpc")

{context['custom_prompt']}
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt),
    ]


def reflexion(context: dict[str, Any]) -> list[Message]:
    sys_prompt = """You are an AI assistant that determines which GCP entities have not been extracted from the given context"""

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

Given the above previous messages, current message, and list of extracted entities; determine if any GCP infrastructure entities haven't been extracted.

Focus on missing:
- Compute Engine instances and compute resources
- VPC networks and networking components
- Storage and database resources
- App Engine and Cloud Run resources
- Security and identity resources
- Messaging and integration services
- Monitoring and management resources
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt),
    ]


def classify_nodes(context: dict[str, Any]) -> list[Message]:
    sys_prompt = """You are an AI assistant that classifies GCP entity nodes given the context from which they were extracted"""

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
    
    Given the above conversation, extracted entities, and provided entity types and their descriptions, classify the extracted GCP entities.
    
    Guidelines:
    1. Each entity must have exactly one type
    2. Only use the provided ENTITY TYPES as types, do not use additional types to classify entities.
    3. If none of the provided entity types accurately classify an extracted node, the type should be set to None
    4. Consider GCP-specific context when classifying (e.g., Compute Engine instances, Cloud Storage buckets, Cloud SQL instances)
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt),
    ]


def extract_attributes(context: dict[str, Any]) -> list[Message]:
    return [
        Message(
            role='system',
            content='You are a helpful assistant that extracts GCP entity properties from the provided text.',
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
        4. Consider GCP-specific attributes like machine types, VPC configurations, storage bucket types, etc.
        
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