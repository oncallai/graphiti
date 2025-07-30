"""
GitHub-specific prompts for node extraction
"""

import json
from typing import Any, Protocol, TypedDict
from pydantic import BaseModel, Field
from ..models import Message, PromptFunction, PromptVersion


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
    sys_prompt = """You are an AI assistant specialized in extracting GitHub repository entities.
    Your primary focus is identifying GitHub repositories, workflows, and development resources from GitHub metadata and code."""

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

You are analyzing GitHub repository metadata and application code. Extract entities focusing on:

## PRIMARY EXTRACTION TARGETS:

1. **GitHub Repositories & Projects**:
   - GitHub repositories (repo names like "user/repo-name")
   - Repository branches (main, develop, feature branches)
   - Repository tags and releases
   - Repository environments (production, staging, development)
   - Repository secrets and variables

2. **GitHub Actions & Workflows**:
   - GitHub Actions workflows (.github/workflows/*.yml)
   - Workflow jobs and steps
   - Self-hosted runners
   - GitHub-hosted runners
   - Action marketplace actions
   - Custom actions

3. **Code & Application Components**:
   - Application names from package.json, requirements.txt, etc.
   - Microservices and services
   - API endpoints and routes
   - Database schemas and models
   - Configuration files and environments
   - Docker containers and images

4. **Dependencies & Libraries**:
   - Package dependencies (npm, pip, maven, etc.)
   - Framework names (React, Django, Spring, etc.)
   - Library names and versions
   - Development tools (webpack, babel, etc.)
   - Testing frameworks (Jest, pytest, etc.)

5. **Infrastructure & Deployment**:
   - Deployment environments (production, staging, dev)
   - Cloud services mentioned in code
   - Database names and types
   - External APIs and services
   - Monitoring and logging services

6. **Development Tools & Services**:
   - CI/CD pipelines (GitHub Actions, Jenkins, etc.)
   - Code quality tools (SonarQube, ESLint, etc.)
   - Documentation platforms (GitHub Pages, ReadTheDocs)
   - Issue tracking systems
   - Code review tools

## GITHUB-SPECIFIC RULES:
- Extract repository names in format "owner/repo-name"
- Include branch names when relevant (e.g., "main", "develop")
- Extract application names from configuration files
- Capture dependency names and versions
- Include environment names (prod, staging, dev)
- Extract from GitHub-specific files (.github/, workflows, etc.)

## GITHUB NAMING CONVENTIONS:
- Repositories: use "owner/repo-name" format
- Branches: use branch names (e.g., "main", "feature/new-feature")
- Applications: use names from config files (e.g., "my-web-app", "api-service")
- Dependencies: use package names (e.g., "react", "django", "express")

## DO NOT EXTRACT:
- Generic file names without context
- Temporary build artifacts
- Configuration values or parameters
- Personal information or tokens
- Generic GitHub service names without specific instances

{context['custom_prompt']}
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt),
    ]


def extract_json(context: dict[str, Any]) -> list[Message]:
    sys_prompt = """You are an AI assistant specialized in extracting GitHub repository entities from JSON configurations.
    Your primary focus is identifying GitHub repositories, workflows, and development resources from GitHub metadata and code."""

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

You are analyzing JSON from GitHub repositories. Extract entities focusing on:

## PRIMARY EXTRACTION TARGETS:

1. **GitHub Repositories & Projects**:
   - GitHub repositories (repo names like "user/repo-name")
   - Repository branches (main, develop, feature branches)
   - Repository tags and releases
   - Repository environments (production, staging, development)
   - Repository secrets and variables

2. **GitHub Actions & Workflows**:
   - GitHub Actions workflows (.github/workflows/*.yml)
   - Workflow jobs and steps
   - Self-hosted runners
   - GitHub-hosted runners
   - Action marketplace actions
   - Custom actions

3. **Code & Application Components**:
   - Application names from package.json, requirements.txt, etc.
   - Microservices and services
   - API endpoints and routes
   - Database schemas and models
   - Configuration files and environments
   - Docker containers and images

4. **Dependencies & Libraries**:
   - Package dependencies (npm, pip, maven, etc.)
   - Framework names (React, Django, Spring, etc.)
   - Library names and versions
   - Development tools (webpack, babel, etc.)
   - Testing frameworks (Jest, pytest, etc.)

5. **Infrastructure & Deployment**:
   - Deployment environments (production, staging, dev)
   - Cloud services mentioned in code
   - Database names and types
   - External APIs and services
   - Monitoring and logging services

6. **Development Tools & Services**:
   - CI/CD pipelines (GitHub Actions, Jenkins, etc.)
   - Code quality tools (SonarQube, ESLint, etc.)
   - Documentation platforms (GitHub Pages, ReadTheDocs)
   - Issue tracking systems
   - Code review tools

## GITHUB-SPECIFIC RULES:
- Extract repository names in format "owner/repo-name"
- Include branch names when relevant (e.g., "main", "develop")
- Extract application names from configuration files
- Capture dependency names and versions
- Include environment names (prod, staging, dev)
- Extract from GitHub-specific files (.github/, workflows, etc.)

## GITHUB NAMING CONVENTIONS:
- Repositories: use "owner/repo-name" format
- Branches: use branch names (e.g., "main", "feature/new-feature")
- Applications: use names from config files (e.g., "my-web-app", "api-service")
- Dependencies: use package names (e.g., "react", "django", "express")

## DO NOT EXTRACT:
- Generic file names without context
- Temporary build artifacts
- Configuration values or parameters
- Personal information or tokens
- Generic GitHub service names without specific instances

{context['custom_prompt']}
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt),
    ]


def extract_text(context: dict[str, Any]) -> list[Message]:
    sys_prompt = """You are an AI assistant specialized in extracting GitHub repository entities from documentation and text files.
    Your primary focus is identifying GitHub repositories, workflows, and development resources from GitHub metadata and code."""

    user_prompt = f"""
<TEXT>
{context['episode_content']}
</TEXT>
<ENTITY TYPES>
{context['entity_types']}
</ENTITY TYPES>

You are analyzing text about GitHub repositories and development. Extract entities focusing on:

## PRIMARY EXTRACTION TARGETS:

1. **GitHub Repositories & Projects**:
   - GitHub repositories (repo names like "user/repo-name")
   - Repository branches (main, develop, feature branches)
   - Repository tags and releases
   - Repository environments (production, staging, development)
   - Repository secrets and variables

2. **GitHub Actions & Workflows**:
   - GitHub Actions workflows (.github/workflows/*.yml)
   - Workflow jobs and steps
   - Self-hosted runners
   - GitHub-hosted runners
   - Action marketplace actions
   - Custom actions

3. **Code & Application Components**:
   - Application names from package.json, requirements.txt, etc.
   - Microservices and services
   - API endpoints and routes
   - Database schemas and models
   - Configuration files and environments
   - Docker containers and images

4. **Dependencies & Libraries**:
   - Package dependencies (npm, pip, maven, etc.)
   - Framework names (React, Django, Spring, etc.)
   - Library names and versions
   - Development tools (webpack, babel, etc.)
   - Testing frameworks (Jest, pytest, etc.)

5. **Infrastructure & Deployment**:
   - Deployment environments (production, staging, dev)
   - Cloud services mentioned in code
   - Database names and types
   - External APIs and services
   - Monitoring and logging services

6. **Development Tools & Services**:
   - CI/CD pipelines (GitHub Actions, Jenkins, etc.)
   - Code quality tools (SonarQube, ESLint, etc.)
   - Documentation platforms (GitHub Pages, ReadTheDocs)
   - Issue tracking systems
   - Code review tools

## GITHUB-SPECIFIC RULES:
- Extract repository names in format "owner/repo-name"
- Include branch names when relevant (e.g., "main", "develop")
- Extract application names from configuration files
- Capture dependency names and versions
- Include environment names (prod, staging, dev)
- Extract from GitHub-specific files (.github/, workflows, etc.)

## GITHUB NAMING CONVENTIONS:
- Repositories: use "owner/repo-name" format
- Branches: use branch names (e.g., "main", "feature/new-feature")
- Applications: use names from config files (e.g., "my-web-app", "api-service")
- Dependencies: use package names (e.g., "react", "django", "express")

## DO NOT EXTRACT:
- Generic file names without context
- Temporary build artifacts
- Configuration values or parameters
- Personal information or tokens
- Generic GitHub service names without specific instances

{context['custom_prompt']}
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt),
    ]


def reflexion(context: dict[str, Any]) -> list[Message]:
    sys_prompt = """You are an AI assistant that determines which GitHub entities have not been extracted from the given context"""

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

Given the above previous messages, current message, and list of extracted entities; determine if any GitHub entities haven't been extracted.

Focus on missing:
- GitHub repositories and branches
- GitHub Actions workflows and jobs
- Application and service names
- Dependencies and frameworks
- Development tools and services
- Infrastructure and deployment resources
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt),
    ]


def classify_nodes(context: dict[str, Any]) -> list[Message]:
    sys_prompt = """You are an AI assistant that classifies GitHub entity nodes given the context from which they were extracted"""

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
    
    Given the above conversation, extracted entities, and provided entity types and their descriptions, classify the extracted GitHub entities.
    
    Guidelines:
    1. Each entity must have exactly one type
    2. Only use the provided ENTITY TYPES as types, do not use additional types to classify entities.
    3. If none of the provided entity types accurately classify an extracted node, the type should be set to None
    4. Consider GitHub-specific context when classifying (e.g., repositories, workflows, applications)
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt),
    ]


def extract_attributes(context: dict[str, Any]) -> list[Message]:
    return [
        Message(
            role='system',
            content='You are a helpful assistant that extracts GitHub entity properties from the provided text.',
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
        4. Consider GitHub-specific attributes like repository URLs, branch information, workflow configurations, etc.
        
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