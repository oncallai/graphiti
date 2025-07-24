"""
CI/CD-specific prompts for node extraction
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
    sys_prompt = """You are an AI assistant specialized in extracting CI/CD pipeline entities.
    Your primary focus is identifying CI/CD tools, build systems, and deployment resources from configuration files and logs."""

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

You are analyzing CI/CD pipeline configurations and logs. Extract entities focusing on:

## PRIMARY EXTRACTION TARGETS:

1. **CI/CD Platforms & Tools**:
   - Jenkins servers and jobs
   - GitHub Actions workflows and runners
   - GitLab CI/CD pipelines and runners
   - Azure DevOps pipelines and agents
   - CircleCI workflows and orbs
   - Travis CI builds and stages
   - TeamCity build configurations
   - Bamboo build plans

2. **Build & Test Systems**:
   - Build tools (Maven, Gradle, npm, yarn, pip, etc.)
   - Testing frameworks (JUnit, Jest, pytest, etc.)
   - Code quality tools (SonarQube, ESLint, etc.)
   - Build servers and agents
   - Build artifacts and packages
   - Test environments and databases

3. **Deployment & Infrastructure**:
   - Deployment environments (production, staging, dev)
   - Container orchestration (Kubernetes, Docker Swarm)
   - Infrastructure as Code tools (Terraform, CloudFormation)
   - Deployment targets (servers, clusters, cloud services)
   - Blue-green deployment environments
   - Canary deployment stages

4. **Artifact & Package Management**:
   - Artifact repositories (Nexus, Artifactory, GitHub Packages)
   - Container registries (Docker Hub, ECR, ACR, GCR)
   - Package managers (npm, Maven, NuGet, PyPI)
   - Build artifacts and binaries
   - Docker images and containers
   - Helm charts and Kubernetes manifests

5. **Monitoring & Observability**:
   - Monitoring tools (Prometheus, Grafana, Datadog)
   - Logging systems (ELK Stack, Splunk, CloudWatch)
   - Alerting systems (PagerDuty, Slack, email)
   - APM tools (New Relic, AppDynamics)
   - Health check endpoints
   - Performance monitoring dashboards

6. **Security & Compliance**:
   - Security scanning tools (SonarQube, Snyk, OWASP ZAP)
   - Vulnerability scanners
   - Compliance checking tools
   - Secret management systems (Vault, AWS Secrets Manager)
   - Code signing tools
   - Security testing environments

7. **Communication & Notifications**:
   - Notification channels (Slack, Teams, email)
   - Chat platforms and webhooks
   - Status pages and dashboards
   - Communication tools for deployments
   - Release notes and changelogs

## CI/CD-SPECIFIC RULES:
- Extract pipeline names and job names
- Include environment names (prod, staging, dev)
- Capture tool and platform names
- Extract from configuration files (YAML, JSON, XML)
- Include version information when relevant
- Capture build and deployment targets

## CI/CD NAMING CONVENTIONS:
- Pipelines: use descriptive names (e.g., "prod-deployment-pipeline", "test-automation")
- Jobs: use specific job names (e.g., "build-and-test", "deploy-to-staging")
- Environments: use environment names (e.g., "production", "staging", "development")
- Tools: use tool names (e.g., "jenkins", "github-actions", "sonarqube")

## DO NOT EXTRACT:
- Generic configuration values or parameters
- Temporary build artifacts or logs
- Personal information or credentials
- Generic service names without specific instances
- Build numbers or commit hashes as separate entities

{context['custom_prompt']}
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt),
    ]


def extract_json(context: dict[str, Any]) -> list[Message]:
    sys_prompt = """You are an AI assistant specialized in extracting CI/CD entities from JSON configurations.
    Focus on pipeline configurations, build files, and CI/CD tool outputs."""

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

You are analyzing JSON from CI/CD systems. This could be:
- Pipeline configuration files
- Build tool outputs
- CI/CD platform API responses
- Deployment manifests
- Configuration management files
- Tool configuration files

## EXTRACTION FOCUS FOR CI/CD JSON:

1. **From Pipeline Configurations**:
   - Pipeline names and job names
   - Stage and step names
   - Environment names and configurations
   - Tool and service names
   - Build and deployment targets

2. **From Build Tool Outputs**:
   - Build tool names and versions
   - Test framework names
   - Artifact names and locations
   - Dependency names and versions
   - Build environment information

3. **From CI/CD Platform APIs**:
   - Pipeline and job identifiers
   - Runner and agent names
   - Environment and deployment names
   - Tool and service configurations
   - Build and test results

4. **From Deployment Manifests**:
   - Deployment environment names
   - Service and application names
   - Infrastructure component names
   - Configuration and secret names
   - Monitoring and logging tools

## CI/CD JSON SPECIFIC RULES:
- Extract pipeline and job names from appropriate fields
- Include environment information when available
- Capture tool and platform names
- Focus on build and deployment targets
- Skip generic configuration values

{context['custom_prompt']}
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt),
    ]


def extract_text(context: dict[str, Any]) -> list[Message]:
    sys_prompt = """You are an AI assistant specialized in extracting CI/CD entities from documentation and logs.
    Focus on CI/CD documentation, deployment guides, and pipeline logs."""

    user_prompt = f"""
<TEXT>
{context['episode_content']}
</TEXT>
<ENTITY TYPES>
{context['entity_types']}
</ENTITY TYPES>

You are analyzing text about CI/CD systems. This could be:
- CI/CD documentation and guides
- Deployment runbooks
- Pipeline logs and outputs
- Build and test reports
- Release notes and changelogs
- Infrastructure documentation

## EXTRACTION FOCUS FOR CI/CD TEXT:

1. **Pipeline Information**:
   - Pipeline names and descriptions
   - Job and stage names
   - Environment names and configurations
   - Tool and platform names
   - Build and deployment targets

2. **Build and Test Components**:
   - Build tool names and versions
   - Test framework names
   - Code quality tool names
   - Artifact and package names
   - Build environment information

3. **Deployment and Infrastructure**:
   - Deployment environment names
   - Infrastructure component names
   - Service and application names
   - Monitoring and logging tools
   - Security and compliance tools

4. **Tools and Platforms**:
   - CI/CD platform names
   - Build and test tool names
   - Deployment tool names
   - Monitoring and alerting tools
   - Communication and notification tools

## CI/CD TEXT SPECIFIC RULES:
- Extract specific pipeline and tool names
- Include environment prefixes/suffixes (prod-, -staging)
- Capture build and deployment target names
- Focus on currently used tools and platforms
- Extract from configuration examples and logs

## CI/CD NAMING STANDARDS:
- Use full pipeline names (e.g., "prod-deployment-pipeline")
- Include environment context (e.g., "staging-test-automation")
- Maintain original naming conventions
- Use canonical names for well-known tools and platforms

{context['custom_prompt']}
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt),
    ]


def reflexion(context: dict[str, Any]) -> list[Message]:
    sys_prompt = """You are an AI assistant that determines which CI/CD entities have not been extracted from the given context"""

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

Given the above previous messages, current message, and list of extracted entities; determine if any CI/CD entities haven't been extracted.

Focus on missing:
- CI/CD platforms and tools
- Build and test systems
- Deployment and infrastructure components
- Artifact and package management tools
- Monitoring and observability tools
- Security and compliance tools
- Communication and notification systems
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt),
    ]


def classify_nodes(context: dict[str, Any]) -> list[Message]:
    sys_prompt = """You are an AI assistant that classifies CI/CD entity nodes given the context from which they were extracted"""

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
    
    Given the above conversation, extracted entities, and provided entity types and their descriptions, classify the extracted CI/CD entities.
    
    Guidelines:
    1. Each entity must have exactly one type
    2. Only use the provided ENTITY TYPES as types, do not use additional types to classify entities.
    3. If none of the provided entity types accurately classify an extracted node, the type should be set to None
    4. Consider CI/CD-specific context when classifying (e.g., pipelines, build tools, deployment environments)
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt),
    ]


def extract_attributes(context: dict[str, Any]) -> list[Message]:
    return [
        Message(
            role='system',
            content='You are a helpful assistant that extracts CI/CD entity properties from the provided text.',
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
        4. Consider CI/CD-specific attributes like pipeline configurations, build environments, deployment targets, etc.
        
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