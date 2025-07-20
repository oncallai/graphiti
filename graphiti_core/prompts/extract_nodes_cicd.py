"""
CI/CD resources specific prompts for node extraction
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
    sys_prompt = """You are an AI assistant specialized in extracting CI/CD pipeline entities.
    Your primary focus is identifying build pipelines, deployment workflows, and automation tools."""

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

You are analyzing CI/CD resources. Extract entities focusing on:

## PRIMARY EXTRACTION TARGETS:

1. **Pipeline Components**:
   - Pipeline names (build, deploy, release)
   - Job/Stage names within pipelines
   - Workflow names (GitHub Actions, GitLab CI)
   - Build configurations
   - Deployment pipelines

2. **CI/CD Infrastructure**:
   - Jenkins instances/controllers
   - GitLab runners
   - GitHub Actions runners
   - Build agents/nodes
   - Artifact repositories (Nexus, Artifactory)

3. **Deployment Targets**:
   - Environment names (dev, staging, prod)
   - Kubernetes clusters for deployments
   - Server groups or deployment targets
   - Container registries (Docker Hub, ECR, ACR)
   - Package registries (npm, PyPI mirrors)

4. **Automation Tools**:
   - Deployment tools (ArgoCD, Flux, Spinnaker)
   - Configuration management (Ansible, Puppet, Chef)
   - Secret managers (Vault, AWS Secrets Manager)
   - Monitoring/alerting integrations

## CI/CD-SPECIFIC RULES:
- Use pipeline names as primary identifiers
- Include environment suffixes (e.g., "deploy-prod", "build-staging")
- Extract tool names and their instances
- Capture integration points with other systems
- Focus on active pipelines, not templates

## NAMING CONVENTIONS:
- Jenkins: job names and folder paths (e.g., "backend/build-job")
- GitLab CI: pipeline names from .gitlab-ci.yml
- GitHub Actions: workflow names from .github/workflows/
- ArgoCD: application names and projects

## DO NOT EXTRACT:
- Individual build numbers or run IDs
- Temporary build artifacts
- Log messages or console output
- Time-based triggers or cron expressions

{context['custom_prompt']}
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt),
    ]


def extract_json(context: dict[str, Any]) -> list[Message]:
    sys_prompt = """You are an AI assistant specialized in extracting CI/CD entities from JSON configurations.
    Focus on pipeline definitions, job configurations, and deployment manifests."""

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

You are analyzing JSON from CI/CD configurations. This could be:
- Jenkins job config.xml as JSON
- GitLab CI pipeline definitions
- GitHub Actions workflow files
- ArgoCD application manifests
- Tekton pipeline resources
- Spinnaker pipeline JSON

## EXTRACTION FOCUS FOR CI/CD JSON:

1. **Pipeline Definitions**:
   - Pipeline names from "name" or "id" fields
   - Stage names from pipeline stages
   - Job names and their types
   - Deployment environments

2. **Infrastructure Resources**:
   - Agent/runner labels or names
   - Artifact repository references
   - Container registry URLs (extract registry name)
   - Tool integrations (SonarQube, etc.)

3. **Deployment Configurations**:
   - Target cluster names
   - Namespace/project names
   - Application names in ArgoCD
   - Environment-specific configurations

## CI/CD JSON SPECIFIC RULES:
- Extract concrete names, not parameter placeholders
- Include environment context in names when present
- Focus on pipeline/job names, not individual steps
- Extract tool names from plugin configurations
- Skip variable definitions unless they name resources

{context['custom_prompt']}
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt),
    ]


def extract_text(context: dict[str, Any]) -> list[Message]:
    sys_prompt = """You are an AI assistant specialized in extracting CI/CD entities from documentation and configuration files.
    Focus on build documentation, deployment guides, and pipeline descriptions."""

    user_prompt = f"""
<TEXT>
{context['episode_content']}
</TEXT>
<ENTITY TYPES>
{context['entity_types']}
</ENTITY TYPES>

You are analyzing text about CI/CD processes. This could be:
- Pipeline documentation
- Deployment runbooks
- CI/CD architecture descriptions
- Build and release notes
- Automation playbooks

## EXTRACTION FOCUS FOR CI/CD TEXT:

1. **Pipeline Names**:
   - Build pipeline names
   - Deployment pipeline names
   - Release workflow names
   - Automated test suite names

2. **Infrastructure Components**:
   - CI/CD tool names (Jenkins, GitLab, etc.)
   - Build server/agent names
   - Artifact repository names
   - Container registries

3. **Deployment Entities**:
   - Environment names and their purposes
   - Deployment target names
   - Application names being deployed
   - Service names in deployments

## CI/CD TEXT SPECIFIC RULES:
- Extract specific pipeline/job names, not generic process descriptions
- Include environment context (e.g., "prod-deployment-pipeline")
- Capture tool-specific terminology correctly
- Focus on automated processes, not manual steps
- Extract integration points between CI/CD tools

## NAMING STANDARDS:
- Use full pipeline names including context
- Maintain tool-specific naming conventions
- Include environment prefixes/suffixes
- Use canonical names for well-known CI/CD tools

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