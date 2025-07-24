"""
Observability Logs-specific prompts for node extraction
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
    sys_prompt = """You are an AI assistant specialized in extracting logging and observability entities.
    Your primary focus is identifying logging systems, log sources, and log management tools from configurations and logs."""

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

You are analyzing logging and observability systems. Extract entities focusing on:

## PRIMARY EXTRACTION TARGETS:

1. **Logging Platforms & Systems**:
   - ELK Stack (Elasticsearch, Logstash, Kibana)
   - Splunk Enterprise and Cloud
   - AWS CloudWatch Logs
   - Azure Monitor Logs
   - Google Cloud Logging
   - Datadog Logs
   - New Relic Logs
   - Fluentd and Fluent Bit
   - Logstash pipelines
   - Graylog

2. **Log Sources & Applications**:
   - Application services and microservices
   - Web servers (Nginx, Apache, IIS)
   - Database systems (PostgreSQL, MySQL, MongoDB)
   - Container platforms (Docker, Kubernetes)
   - Cloud services and APIs
   - Infrastructure components (servers, load balancers)
   - Security systems and firewalls
   - Network devices and routers

3. **Log Management & Processing**:
   - Log aggregators and collectors
   - Log parsers and processors
   - Log routing and forwarding systems
   - Log retention and archival systems
   - Log backup and recovery systems
   - Log search and indexing engines
   - Log analytics and reporting tools

4. **Log Storage & Infrastructure**:
   - Log storage systems and databases
   - Log file systems and volumes
   - Log backup storage (S3, Azure Blob, GCS)
   - Log archival systems
   - Log compression and deduplication tools
   - Log encryption and security tools

5. **Log Analysis & Visualization**:
   - Log analysis dashboards
   - Log visualization tools
   - Log reporting systems
   - Log alerting and notification systems
   - Log correlation and analysis tools
   - Log machine learning and AI tools

6. **Log Security & Compliance**:
   - Log security monitoring tools
   - Log compliance and audit systems
   - Log access control and authentication
   - Log encryption and data protection
   - Log retention policies and tools
   - Log forensic analysis tools

7. **Log Integration & APIs**:
   - Log API endpoints and services
   - Log integration platforms
   - Log webhook and notification systems
   - Log export and import tools
   - Log synchronization systems
   - Log federation and sharing tools

## LOGGING-SPECIFIC RULES:
- Extract logging platform and system names
- Include log source application and service names
- Capture log management tool names
- Extract from logging configurations and manifests
- Include environment names (prod, staging, dev)
- Capture log storage and infrastructure names

## LOGGING NAMING CONVENTIONS:
- Logging platforms: use platform names (e.g., "elasticsearch-cluster", "splunk-enterprise")
- Log sources: use application/service names (e.g., "web-app-logs", "api-service-logs")
- Log storage: use storage system names (e.g., "log-storage-s3", "log-backup-azure")
- Log dashboards: use dashboard names (e.g., "application-logs-dashboard", "error-logs-view")

## DO NOT EXTRACT:
- Generic log file names without context
- Temporary log files or artifacts
- Log configuration values or parameters
- Personal information or credentials
- Generic logging terms without specific instances

{context['custom_prompt']}
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt),
    ]


def extract_json(context: dict[str, Any]) -> list[Message]:
    sys_prompt = """You are an AI assistant specialized in extracting logging entities from JSON configurations.
    Focus on logging platform configurations, log source definitions, and log management tool settings."""

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

You are analyzing JSON from logging systems. This could be:
- Logging platform configurations
- Log source definitions
- Log management tool settings
- Log storage configurations
- Log analysis dashboard definitions
- Log integration API responses

## EXTRACTION FOCUS FOR LOGGING JSON:

1. **From Logging Platform Configs**:
   - Platform names and cluster names
   - Index and index pattern names
   - Pipeline and processor names
   - Dashboard and visualization names
   - Alert and notification names

2. **From Log Source Definitions**:
   - Application and service names
   - Log file paths and names
   - Container and pod names
   - Infrastructure component names
   - Log stream and topic names

3. **From Log Management Tools**:
   - Tool and system names
   - Configuration and policy names
   - Storage and backup system names
   - Integration and API endpoint names
   - Security and compliance tool names

4. **From Log Analysis Configs**:
   - Dashboard and report names
   - Query and filter names
   - Alert and notification names
   - Visualization and chart names
   - Export and integration names

## LOGGING JSON SPECIFIC RULES:
- Extract logging platform names from appropriate fields
- Include log source application and service names
- Capture log management tool and system names
- Focus on log storage and infrastructure names
- Skip generic configuration values

{context['custom_prompt']}
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt),
    ]


def extract_text(context: dict[str, Any]) -> list[Message]:
    sys_prompt = """You are an AI assistant specialized in extracting logging entities from documentation and logs.
    Focus on logging documentation, log analysis reports, and logging system descriptions."""

    user_prompt = f"""
<TEXT>
{context['episode_content']}
</TEXT>
<ENTITY TYPES>
{context['entity_types']}
</ENTITY TYPES>

You are analyzing text about logging systems. This could be:
- Logging system documentation
- Log analysis reports
- Logging configuration guides
- Log monitoring dashboards
- Log troubleshooting guides
- Logging architecture documentation

## EXTRACTION FOCUS FOR LOGGING TEXT:

1. **Logging Platform Information**:
   - Platform names and descriptions
   - Cluster and system names
   - Index and pattern names
   - Dashboard and visualization names
   - Alert and notification names

2. **Log Source Information**:
   - Application and service names
   - Infrastructure component names
   - Log file and stream names
   - Container and pod names
   - Network and security device names

3. **Log Management Information**:
   - Tool and system names
   - Storage and backup system names
   - Processing and analysis tool names
   - Integration and API names
   - Security and compliance tool names

4. **Log Analysis Information**:
   - Dashboard and report names
   - Query and filter names
   - Alert and notification names
   - Export and integration names
   - Visualization and chart names

## LOGGING TEXT SPECIFIC RULES:
- Extract specific logging platform and tool names
- Include environment prefixes/suffixes (prod-, -staging)
- Capture log source application and service names
- Focus on currently used logging tools and platforms
- Extract from configuration examples and logs

## LOGGING NAMING STANDARDS:
- Use full logging platform names (e.g., "elasticsearch-production-cluster")
- Include environment context (e.g., "staging-application-logs")
- Maintain original naming conventions
- Use canonical names for well-known logging platforms

{context['custom_prompt']}
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt),
    ]


def reflexion(context: dict[str, Any]) -> list[Message]:
    sys_prompt = """You are an AI assistant that determines which logging entities have not been extracted from the given context"""

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

Given the above previous messages, current message, and list of extracted entities; determine if any logging entities haven't been extracted.

Focus on missing:
- Logging platforms and systems
- Log sources and applications
- Log management and processing tools
- Log storage and infrastructure
- Log analysis and visualization tools
- Log security and compliance tools
- Log integration and API systems
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt),
    ]


def classify_nodes(context: dict[str, Any]) -> list[Message]:
    sys_prompt = """You are an AI assistant that classifies logging entity nodes given the context from which they were extracted"""

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
    
    Given the above conversation, extracted entities, and provided entity types and their descriptions, classify the extracted logging entities.
    
    Guidelines:
    1. Each entity must have exactly one type
    2. Only use the provided ENTITY TYPES as types, do not use additional types to classify entities.
    3. If none of the provided entity types accurately classify an extracted node, the type should be set to None
    4. Consider logging-specific context when classifying (e.g., logging platforms, log sources, log management tools)
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt),
    ]


def extract_attributes(context: dict[str, Any]) -> list[Message]:
    return [
        Message(
            role='system',
            content='You are a helpful assistant that extracts logging entity properties from the provided text.',
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
        4. Consider logging-specific attributes like log formats, retention policies, storage configurations, etc.
        
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