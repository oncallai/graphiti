"""
Application-centric prompts for cross-domain node extraction
This prompt identifies main application entities that serve as central hubs
connecting resources across GitHub, Cloud, CI/CD, and Observability domains.
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
    sys_prompt = """You are an AI assistant specialized in extracting main application entities that serve as central hubs.
    Your primary focus is identifying the core applications that connect resources across GitHub, Cloud, CI/CD, and Observability domains."""

    user_prompt = f"""
<PREVIOUS_MESSAGES>
{json.dumps([ep for ep in context['previous_episodes']], indent=2)}
</PREVIOUS_MESSAGES>

<CURRENT_MESSAGE>
{context['episode_content']}
</CURRENT_MESSAGE>

<ENTITY TYPES>
{context['entity_types']}
</ENTITY TYPES>

You are analyzing cross-domain application resources. Extract entities focusing on:

## PRIMARY EXTRACTION TARGETS:

1. **Main Applications** (Central Hubs):
   - Primary application names (e-commerce-platform, user-service, analytics-platform)
   - Core business applications
   - Microservice applications
   - Platform applications
   - System applications

2. **Application Variants** (Environment/Deployment):
   - Production applications (e-commerce-platform-prod)
   - Staging applications (user-service-staging)
   - Development applications (analytics-platform-dev)
   - Regional deployments (e-commerce-platform-us-east-1)

3. **Application Components** (Sub-Services):
   - Frontend applications (e-commerce-frontend)
   - Backend APIs (e-commerce-api)
   - Worker services (e-commerce-worker)
   - Background jobs (e-commerce-scheduler)

## APPLICATION-CENTRIC EXTRACTION RULES:

1. **Identify Core Applications**: Look for the main application names that appear across multiple domains
2. **Cross-Domain Consistency**: Extract applications that are referenced in GitHub, Cloud, CI/CD, and Observability contexts
3. **Naming Patterns**: Look for consistent naming patterns across domains
4. **Business Context**: Focus on applications that represent business capabilities
5. **Central Hub Potential**: Extract applications that can serve as central nodes connecting all resources

## NAMING CONVENTIONS:

### **Main Applications**:
- Use canonical application names (e.g., "e-commerce-platform", "user-service")
- Include business context in names
- Use consistent naming across domains
- Avoid environment-specific suffixes for main applications

### **Application Variants**:
- Include environment context (e.g., "e-commerce-platform-prod")
- Include region/zone information when relevant
- Use deployment-specific naming

## DO EXTRACT:
- Main application names that appear across domains
- Core business applications
- Microservice application names
- Platform application names
- Application names from repository names
- Service names from deployment configurations
- Application names from monitoring configurations

## DO NOT EXTRACT:
- Individual resource names (databases, caches, etc.)
- Infrastructure components (load balancers, VPCs, etc.)
- Tool names (Jenkins, Prometheus, etc.)
- Generic terms without specific application context
- Temporary or build-specific names

{context['custom_prompt']}
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt),
    ]


def extract_json(context: dict[str, Any]) -> list[Message]:
    sys_prompt = """You are an AI assistant specialized in extracting main application entities from JSON configurations.
    Focus on application names that appear across multiple domain configurations."""

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

You are analyzing JSON from cross-domain configurations. This could be:
- Application configuration files
- Deployment manifests
- Service definitions
- Cross-domain integration configs
- Application metadata

## EXTRACTION FOCUS FOR APPLICATION JSON:

1. **From Application Configs**:
   - Application names from "name", "application_name", "service_name" fields
   - Main service identifiers
   - Application identifiers across domains

2. **From Deployment Manifests**:
   - Application names in deployment configurations
   - Service names in orchestration files
   - Application identifiers in container configs

3. **From Cross-Domain Configs**:
   - Application names that appear in multiple domain contexts
   - Central application identifiers
   - Business application names

## APPLICATION JSON SPECIFIC RULES:
- Extract main application names, not individual resources
- Focus on names that could serve as central hubs
- Look for consistent naming across different JSON structures
- Extract application names that connect multiple domains
- Focus on business application names, not technical resource names

{context['custom_prompt']}
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt),
    ]


def extract_text(context: dict[str, Any]) -> list[Message]:
    sys_prompt = """You are an AI assistant specialized in extracting main application entities from documentation and descriptions.
    Focus on applications that serve as central hubs connecting resources across domains."""

    user_prompt = f"""
<TEXT>
{context['episode_content']}
</TEXT>
<ENTITY TYPES>
{context['entity_types']}
</ENTITY TYPES>

You are analyzing text about cross-domain applications. This could be:
- Application architecture documentation
- System overview descriptions
- Cross-domain integration guides
- Application deployment documentation
- Business application descriptions

## EXTRACTION FOCUS FOR APPLICATION TEXT:

1. **Main Applications**:
   - Core business application names
   - Primary service applications
   - Platform application names
   - System application names

2. **Cross-Domain Applications**:
   - Applications mentioned across multiple domains
   - Central application hubs
   - Business capability applications
   - Integration point applications

3. **Application Context**:
   - Applications that connect multiple resources
   - Central applications in system architecture
   - Business applications with multiple dependencies

## APPLICATION TEXT SPECIFIC RULES:
- Extract main application names, not individual components
- Focus on applications that can serve as central hubs
- Look for applications mentioned across multiple contexts
- Extract business application names, not technical resource names
- Focus on applications that connect multiple domains

## NAMING STANDARDS:
- Use canonical application names
- Include business context when relevant
- Maintain consistency with cross-domain naming
- Use full application names, not abbreviations

{context['custom_prompt']}
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt),
    ]


def reflexion(context: dict[str, Any]) -> list[Message]:
    sys_prompt = """You are an AI assistant that determines which main application entities have not been extracted from the given context"""

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

Given the above previous messages, current message, and list of extracted entities; determine if any main application entities haven't been
extracted.
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt),
    ]


def classify_nodes(context: dict[str, Any]) -> list[Message]:
    sys_prompt = """You are an AI assistant that classifies application entity nodes given the context from which they were extracted"""

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
    
    Given the above conversation, extracted entities, and provided entity types and their descriptions, classify the extracted application entities.
    
    Guidelines:
    1. Each entity must have exactly one type
    2. Only use the provided ENTITY TYPES as types, do not use additional types to classify entities.
    3. If none of the provided entity types accurately classify an extracted node, the type should be set to None
    4. Focus on classifying main application entities that can serve as central hubs
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt),
    ]


def extract_attributes(context: dict[str, Any]) -> list[Message]:
    return [
        Message(
            role='system',
            content='You are a helpful assistant that extracts application entity properties from the provided text.',
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
        4. Focus on application-specific attributes like business context, cross-domain connections, and central hub characteristics.
        
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