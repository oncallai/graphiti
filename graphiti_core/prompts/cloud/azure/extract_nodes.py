"""
Azure-specific prompts for node extraction
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
    sys_prompt = """You are an AI assistant specialized in extracting Azure infrastructure entities.
    Your primary focus is identifying Azure resources, their configurations, and relationships from Azure configurations and IaC files."""

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

You are analyzing Azure infrastructure resources. Extract entities focusing on:

## PRIMARY EXTRACTION TARGETS:

1. **Virtual Machines & Compute Resources**:
   - Virtual Machines (VM names)
   - Virtual Machine Scale Sets (VMSS)
   - Container Instances and Container Groups
   - Azure Kubernetes Service (AKS) clusters
   - Azure Functions
   - Batch accounts and pools

2. **Virtual Networks & Networking Resources**:
   - Virtual Networks (VNet names)
   - Subnets (with subnet names)
   - Network Security Groups (NSG)
   - Route Tables
   - Application Gateways
   - Load Balancers
   - Virtual Network Gateways
   - ExpressRoute circuits

3. **Storage & Database Resources**:
   - Storage Accounts
   - Blob containers and file shares
   - Managed Disks
   - Azure SQL Database instances
   - Azure Database for PostgreSQL/MySQL
   - Cosmos DB accounts
   - Redis Cache instances
   - Data Lake Storage accounts

4. **App Service & Web Resources**:
   - App Service Plans
   - Web Apps and API Apps
   - Function Apps
   - Static Web Apps
   - App Service Environments (ASE)
   - Deployment slots

5. **Security & Identity**:
   - Azure Active Directory (Azure AD) applications
   - Managed Identities
   - Key Vaults and secrets
   - Role Definitions and Assignments
   - Application Insights
   - Azure Security Center resources

6. **Messaging & Integration**:
   - Service Bus namespaces and queues
   - Event Hubs namespaces
   - Event Grid topics and subscriptions
   - Logic Apps workflows
   - API Management services

7. **Monitoring & Management**:
   - Log Analytics workspaces
   - Application Insights resources
   - Azure Monitor action groups
   - Azure Backup vaults
   - Recovery Services vaults

## AZURE-SPECIFIC RULES:
- Use resource names as primary identifiers (Azure uses names rather than IDs)
- Include resource group information when relevant
- Include region information in entity names when relevant (e.g., "prod-web-app-eastus")
- Extract Managed Identities and service principals as entities
- Capture resource tags that indicate ownership or purpose
- Include environment prefixes (prod-, staging-, dev-)

## AZURE NAMING CONVENTIONS:
- Virtual Machines: use VM names (e.g., "prodWebVM", "stagingAppServer")
- Virtual Networks: use VNet names (e.g., "prodVNet", "stagingNetwork")
- Storage Accounts: use storage account names (e.g., "myappstorage", "prodlogs")
- App Services: use app names (e.g., "myWebApp", "prodAPI")

## DO NOT EXTRACT:
- IP addresses or CIDR blocks as separate entities
- Temporary resources or build artifacts
- Configuration values or parameters
- Cost information or billing details
- Generic Azure service names without specific instances

{context['custom_prompt']}
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt),
    ]


def extract_json(context: dict[str, Any]) -> list[Message]:
    sys_prompt = """You are an AI assistant specialized in extracting Azure infrastructure entities from JSON configurations.
    Focus on ARM templates, Terraform state files, Azure CLI output, and Azure API responses."""

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

You are analyzing JSON from Azure infrastructure. This could be:
- ARM templates
- Terraform state files (.tfstate)
- Azure CLI output (az vm list, etc.)
- Azure API responses
- Bicep templates
- Azure Resource Graph queries

## EXTRACTION FOCUS FOR AZURE JSON:

1. **From ARM Templates**:
   - Resource names from "resources" array
   - Resource types (Microsoft.Compute/virtualMachines, etc.)
   - Resource group names
   - Output values representing important resources

2. **From Terraform State**:
   - Resource names from "resources" array
   - Resource types and their identifiers
   - Module names representing infrastructure components
   - Azure-specific resource attributes

3. **From Azure CLI Output**:
   - VM names from vm list
   - VNet names from network vnet list
   - Storage account names from storage account list
   - App service names from webapp list
   - Resource group names from group list

4. **From Azure API Responses**:
   - Resource identifiers and names
   - Service-specific resource types
   - Tagged resources with meaningful names

## AZURE JSON SPECIFIC RULES:
- Extract resource names from appropriate fields (name, id, resourceId)
- Include resource type in extraction when it helps identify the entity
- For nested resources, maintain the hierarchy in naming
- Skip parameter definitions and variable declarations
- Focus on actual provisioned resources, not templates
- Extract from both "resources" and "outputs" sections in ARM templates

{context['custom_prompt']}
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt),
    ]


def extract_text(context: dict[str, Any]) -> list[Message]:
    sys_prompt = """You are an AI assistant specialized in extracting Azure infrastructure entities from documentation and configuration files.
    Focus on Azure infrastructure documentation, runbooks, and deployment guides."""

    user_prompt = f"""
<TEXT>
{context['episode_content']}
</TEXT>
<ENTITY TYPES>
{context['entity_types']}
</ENTITY TYPES>

You are analyzing text about Azure infrastructure. This could be:
- Azure infrastructure documentation
- Deployment guides and runbooks
- Architecture diagrams descriptions
- Azure Well-Architected reviews
- Cloud migration plans
- Azure service documentation

## EXTRACTION FOCUS FOR AZURE TEXT:

1. **Infrastructure Components**:
   - VNet names and network segments
   - VM instance groups or VMSS
   - SQL Database cluster names and instances
   - Storage account names
   - Load balancer names (ALB, Application Gateway)

2. **Azure Managed Services**:
   - AKS cluster names
   - App Service names and plans
   - Function App names
   - API Management service names
   - CDN profile names
   - Service Bus namespace names

3. **Architecture Entities**:
   - Environment names (prod, staging, dev)
   - Application tier names (web, app, data)
   - Azure regions or availability zones
   - Disaster recovery sites
   - Azure subscription names or IDs

## AZURE TEXT SPECIFIC RULES:
- Extract specific resource names, not generic descriptions
- Include environment prefixes/suffixes (prod-, -staging)
- Capture multi-region deployments as separate entities
- Focus on currently deployed resources, not planned ones
- Extract from architecture diagrams if textually described
- Include Azure service names when they represent specific instances

## AZURE NAMING STANDARDS:
- Use full resource names including environment (e.g., "prodWebApp")
- Include region in name if multi-region (e.g., "cacheClusterEastUS")
- Maintain Azure naming conventions (PascalCase, camelCase)
- Use canonical names for well-known Azure services
- Include subscription context when relevant (e.g., "prodSubscriptionVNet")

{context['custom_prompt']}
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt),
    ]


def reflexion(context: dict[str, Any]) -> list[Message]:
    sys_prompt = """You are an AI assistant that determines which Azure entities have not been extracted from the given context"""

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

Given the above previous messages, current message, and list of extracted entities; determine if any Azure infrastructure entities haven't been extracted.

Focus on missing:
- Virtual Machines and compute resources
- Virtual Networks and networking components
- Storage and database resources
- App Services and web resources
- Security and identity resources
- Messaging and integration services
- Monitoring and management resources
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt),
    ]


def classify_nodes(context: dict[str, Any]) -> list[Message]:
    sys_prompt = """You are an AI assistant that classifies Azure entity nodes given the context from which they were extracted"""

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
    
    Given the above conversation, extracted entities, and provided entity types and their descriptions, classify the extracted Azure entities.
    
    Guidelines:
    1. Each entity must have exactly one type
    2. Only use the provided ENTITY TYPES as types, do not use additional types to classify entities.
    3. If none of the provided entity types accurately classify an extracted node, the type should be set to None
    4. Consider Azure-specific context when classifying (e.g., Virtual Machines, Storage Accounts, App Services)
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt),
    ]


def extract_attributes(context: dict[str, Any]) -> list[Message]:
    return [
        Message(
            role='system',
            content='You are a helpful assistant that extracts Azure entity properties from the provided text.',
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
        4. Consider Azure-specific attributes like VM sizes, VNet configurations, storage account types, etc.
        
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