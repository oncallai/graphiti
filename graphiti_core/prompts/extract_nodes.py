"""
Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
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
    sys_prompt = """You are an AI assistant specialized in extracting application entities and infrastructure dependencies from GitHub repository metadata and application code. 
    Your primary task is to identify applications and their infrastructure components (databases, caches, message queues, etc.)."""

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

Instructions:

You are analyzing GitHub repository metadata and application code. Your task is to extract **ONLY** application entities and infrastructure dependency entities from the CURRENT MESSAGE.

## EXTRACTION FOCUS:

### 1. **Application Entities** - Extract these types:
   - Application names (services, microservices, web apps, APIs)
   - Software projects or modules
   - GitHub repositories (when they represent applications)
   - Software components or systems

### 2. **Infrastructure Dependencies** - Extract these types:
   - **Databases**: PostgreSQL, MySQL, MongoDB, Redis, etc.
   - **Caches**: Redis, Memcached, etc.
   - **Message Queues**: RabbitMQ, Kafka, SQS, etc.
   - **Storage Systems**: S3, MinIO, etc.
   - **External Services**: Third-party APIs, cloud services
   - **Infrastructure Components**: Load balancers, proxies, etc.

## EXTRACTION RULES:

### **DO EXTRACT**:
- Application names from repository names, package.json, requirements.txt, etc.
- Database names, connection strings, or database references
- Cache names or Redis instance references
- Service names from docker-compose.yml, kubernetes manifests
- Infrastructure component names from configuration files
- Dependency names from package managers (npm, pip, maven, etc.)

### **DO NOT EXTRACT**:
- People, users, developers, or team names
- Companies or organizations (unless they are infrastructure services)
- Dates, times, versions, or temporal information
- File names, directory paths, or code snippets
- Programming languages, frameworks (unless they are the main application)
- Generic terms like "API", "service", "database" without specific names
- Configuration parameters, environment variables, or settings
- Documentation, comments, or descriptive text

## NAMING CONVENTIONS:

### **Applications**:
- Use the actual application/service name (e.g., "user-service", "e-commerce-platform")
- For GitHub repos, use the repository name if it represents an application
- Use descriptive names that identify the specific application

### **Infrastructure**:
- Use specific instance names when available (e.g., "users_db", "session_cache")
- For generic references, use the service type + context (e.g., "postgres_main", "redis_cache")
- Maintain consistency with naming patterns in the codebase

## ENTITY CLASSIFICATION:
- Use the descriptions in ENTITY TYPES to classify each extracted entity
- Assign the appropriate `entity_type_id` for each application or infrastructure component
- Ensure infrastructure dependencies are properly classified by their type

## QUALITY REQUIREMENTS:
- Extract only entities that are **explicitly mentioned** in the CURRENT MESSAGE
- Use **exact names** from the source material - do not modify or interpret
- Ensure entity names are **specific and meaningful**
- Avoid generic or placeholder names
- Focus on **concrete, identifiable** applications and infrastructure components

{context['custom_prompt']}
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt),
    ]


def extract_json(context: dict[str, Any]) -> list[Message]:
    sys_prompt = """You are an AI assistant specialized in extracting application entities and infrastructure dependencies from JSON data. 
    Your primary task is to identify applications and their infrastructure components from JSON configuration files, package manifests, and metadata."""

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

Instructions:

You are analyzing JSON data that may contain application metadata, configuration files, package manifests, or infrastructure definitions. Your task is to extract **ONLY** application entities and infrastructure dependency entities from the JSON.

## EXTRACTION FOCUS:

### 1. **Application Entities** - Extract from JSON fields like:
   - Application names from "name", "application_name", "service_name" fields
   - Repository names from "repository", "repo", "github_repo" fields
   - Project names from package.json, composer.json, pom.xml equivalents
   - Service definitions from docker-compose.yml, kubernetes manifests
   - Module or component names from configuration files

### 2. **Infrastructure Dependencies** - Extract from JSON fields like:
   - **Database references**: "database", "db_name", "postgres", "mongodb", etc.
   - **Cache references**: "redis", "cache", "memcached", etc.
   - **Message Queue references**: "rabbitmq", "kafka", "sqs", etc.
   - **Storage references**: "s3", "storage", "bucket", etc.
   - **External Service references**: API endpoints, cloud services
   - **Dependencies**: from "dependencies", "requires", "imports" sections

## JSON EXTRACTION RULES:

### **DO EXTRACT**:
- Values from fields that represent application or service names
- Database connection strings or database names
- Cache service names or Redis instance references
- Infrastructure service names from configuration
- Dependency names from package manager files
- Service names from orchestration files (docker-compose, k8s)
- External service references or API names

### **DO NOT EXTRACT**:
- Version numbers, timestamps, or dates
- Configuration parameters, environment variables
- File paths, URLs, or connection strings (extract only the service name)
- Generic field names or JSON keys
- Metadata like "created_at", "updated_at", "version"
- User information, team names, or personal data
- Documentation fields, descriptions, or comments

## NAMING CONVENTIONS:

### **Applications**:
- Use the exact value from name fields (e.g., "user-service", "payment-api")
- For repository references, use the repo name if it represents an application
- Maintain original casing and formatting when meaningful

### **Infrastructure**:
- Use specific instance names when available (e.g., "users_db", "session_cache")
- Extract service type from connection strings (e.g., "postgresql" from postgres://...)
- Use canonical service names for well-known infrastructure (e.g., "redis", "mongodb")

## ENTITY CLASSIFICATION:
- Use the descriptions in ENTITY TYPES to classify each extracted entity
- Assign the appropriate `entity_type_id` for each application or infrastructure component
- Ensure infrastructure dependencies are properly classified by their type

## QUALITY REQUIREMENTS:
- Extract only entities that are **explicitly present** in the JSON values
- Use **exact values** from JSON fields - do not modify or interpret
- Focus on **concrete, identifiable** applications and infrastructure components
- Avoid extracting JSON keys unless they represent entity names
- Ensure extracted names are **specific and meaningful**

{context['custom_prompt']}
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt),
    ]


def extract_text(context: dict[str, Any]) -> list[Message]:

    sys_prompt = """You are an AI assistant specialized in extracting application entities and infrastructure dependencies from text data. 
    Your primary task is to identify applications and their infrastructure components from documentation, code comments, README files, and technical descriptions."""

    user_prompt = f"""
<TEXT>
{context['episode_content']}
</TEXT>
<ENTITY TYPES>
{context['entity_types']}
</ENTITY TYPES>

Instructions:

You are analyzing text that may contain technical documentation, code comments, README files, or descriptions of software systems. Your task is to extract **ONLY** application entities and infrastructure dependency entities from the TEXT.

## EXTRACTION FOCUS:

### 1. **Application Entities** - Extract these types:
   - Application names, service names, microservices
   - Software projects, modules, or components
   - System names or platform names
   - API names or web service names
   - Repository names (when they represent applications)

### 2. **Infrastructure Dependencies** - Extract these types:
   - **Databases**: PostgreSQL, MySQL, MongoDB, Redis, etc.
   - **Caches**: Redis, Memcached, etc.
   - **Message Queues**: RabbitMQ, Kafka, SQS, etc.
   - **Storage Systems**: S3, MinIO, file systems, etc.
   - **External Services**: Third-party APIs, cloud services
   - **Infrastructure Components**: Load balancers, proxies, etc.

## TEXT EXTRACTION RULES:

### **DO EXTRACT**:
- Specific application or service names mentioned in the text
- Database names or database system references
- Cache service names or caching system references
- Message queue or event streaming system names
- Storage service names or storage system references
- External API names or third-party service references
- Infrastructure component names from technical descriptions

### **DO NOT EXTRACT**:
- People, users, developers, or team names
- Companies or organizations (unless they are infrastructure services)
- Dates, times, versions, or temporal information
- File names, directory paths, or code snippets
- Programming languages or frameworks (unless they are the main application)
- Generic terms like "API", "service", "database" without specific names
- Configuration parameters, environment variables
- Documentation sections, headers, or descriptive text

## NAMING CONVENTIONS:

### **Applications**:
- **Custom Applications**: Use the complete, precise name as written in the text
  - Example: "UserManagementService" → "UserManagementService"
  - Example: "payment-processing-api" → "payment-processing-api"
  - Example: "E-Commerce Platform" → "E-Commerce Platform"

### **Infrastructure**:
- **Third-Party Services**: Use canonical service names (lowercase, standardized)
  - Example: "PostgreSQL Database" → "postgresql"
  - Example: "Redis Cache" → "redis"
  - Example: "AWS S3 Storage" → "s3"
  - Example: "MongoDB Database" → "mongodb"
  - Example: "RabbitMQ Message Broker" → "rabbitmq"
  - Example: "Elasticsearch Service" → "elasticsearch"

- **Cloud Services**: Use standard service identifiers
  - Example: "AWS Lambda Function" → "lambda"
  - Example: "Azure Blob Storage" → "blob-storage"
  - Example: "Google Cloud Pub/Sub" → "pubsub"
  - Example: "AWS EC2 Instance" → "ec2"

- **Specific Instances**: Use instance names when available
  - Example: "users_db database" → "users_db"
  - Example: "session_cache Redis" → "session_cache"

## ENTITY CLASSIFICATION:
- Use the descriptions in ENTITY TYPES to classify each extracted entity
- Assign the appropriate `entity_type_id` for each application or infrastructure component
- Ensure infrastructure dependencies are properly classified by their type

## QUALITY REQUIREMENTS:
- Extract only entities that are **explicitly mentioned** in the TEXT
- Use **exact names** from the text - do not modify or interpret unnecessarily
- Focus on **concrete, identifiable** applications and infrastructure components
- Avoid generic or placeholder names
- Ensure entity names are **specific and meaningful**
- Maintain consistency with established naming conventions
- Verify that extracted names are complete and accurate

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
