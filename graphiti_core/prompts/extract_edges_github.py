"""
GitHub repository specific prompts for edge extraction
"""

import json
from typing import Any, Protocol, TypedDict
from pydantic import BaseModel, Field
from .models import Message, PromptFunction, PromptVersion


class Edge(BaseModel):
    relation_type: str = Field(..., description='FACT_PREDICATE_IN_SCREAMING_SNAKE_CASE')
    source_entity_id: int = Field(..., description='The id of the source entity of the fact.')
    target_entity_id: int = Field(..., description='The id of the target entity of the fact.')
    fact: str = Field(..., description='')
    valid_at: str | None = Field(
        None,
        description='The date and time when the relationship described by the edge fact became true or was established. Use ISO 8601 format (YYYY-MM-DDTHH:MM:SS.SSSSSSZ)',
    )
    invalid_at: str | None = Field(
        None,
        description='The date and time when the relationship described by the edge fact stopped being true or ended. Use ISO 8601 format (YYYY-MM-DDTHH:MM:SS.SSSSSSZ)',
    )


class ExtractedEdges(BaseModel):
    edges: list[Edge]


class MissingFacts(BaseModel):
    missing_facts: list[str] = Field(..., description="facts that weren't extracted")


class Prompt(Protocol):
    edge: PromptVersion
    reflexion: PromptVersion
    extract_attributes: PromptVersion


class Versions(TypedDict):
    edge: PromptFunction
    reflexion: PromptFunction
    extract_attributes: PromptFunction


def edge(context: dict[str, Any]) -> list[Message]:
    return [
        Message(
            role='system',
            content='You are an expert at extracting relationships between software components from GitHub repositories. '
            'Focus on code dependencies, service connections, and infrastructure relationships.',
        ),
        Message(
            role='user',
            content=f"""
<PREVIOUS_MESSAGES>
{json.dumps([ep for ep in context['previous_episodes']], indent=2)}
</PREVIOUS_MESSAGES>

<CURRENT_MESSAGE>
{context['episode_content']}
</CURRENT_MESSAGE>

<ENTITIES>
{context['nodes']} 
</ENTITIES>

<REFERENCE_TIME>
{context['reference_time']}
</REFERENCE_TIME>

<FACT TYPES>
{context['edge_types']}
</FACT TYPES>

# TASK
Extract relationships between entities specific to GitHub repositories and software development.

# GITHUB-SPECIFIC RELATIONSHIPS TO EXTRACT:

1. **Code Dependencies**:
   - DEPENDS_ON: Package/library dependencies (app DEPENDS_ON library)
   - IMPORTS: Module imports (service IMPORTS module)
   - USES_FRAMEWORK: Application uses framework (app USES_FRAMEWORK express)
   - REQUIRES: Build/runtime requirements

2. **Service Connections**:
   - CONNECTS_TO: Service connections (api CONNECTS_TO database)
   - QUERIES: Database queries (service QUERIES postgres)
   - CACHES_IN: Caching relationships (api CACHES_IN redis)
   - PUBLISHES_TO: Message publishing (service PUBLISHES_TO rabbitmq)
   - CONSUMES_FROM: Message consumption (worker CONSUMES_FROM kafka)

3. **Repository Structure**:
   - CONTAINS: Repository contains services (repo CONTAINS microservice)
   - PART_OF: Component hierarchy (module PART_OF application)
   - EXPOSES: API exposure (service EXPOSES rest_api)
   - IMPLEMENTS: Interface implementation (service IMPLEMENTS api_spec)

4. **Deployment Relationships**:
   - DEPLOYED_WITH: Co-deployment (service DEPLOYED_WITH database)
   - RUNS_IN: Container relationships (app RUNS_IN docker_container)
   - COMPOSED_IN: Docker compose services (service COMPOSED_IN docker_compose)
   - CONFIGURED_BY: Configuration sources (app CONFIGURED_BY config_file)

5. **Development Workflow**:
   - BUILT_BY: Build relationships (app BUILT_BY github_action)
   - TESTED_BY: Testing relationships (service TESTED_BY test_suite)
   - PACKAGED_AS: Packaging relationships (app PACKAGED_AS docker_image)

# EXTRACTION RULES FOR GITHUB:

1. Focus on technical relationships found in code, configuration files, and documentation
2. Use CONNECTS_TO for any service-to-service communication
3. Use DEPENDS_ON for compile-time dependencies
4. Use REQUIRES for runtime dependencies
5. Extract from:
   - package.json dependencies
   - docker-compose.yml service definitions
   - Connection strings in code
   - Import statements
   - Configuration files
   - CI/CD workflows

{context['custom_prompt']}

# EXAMPLES:
- If code shows "const db = new Pool({host: 'postgres'})", extract: service CONNECTS_TO postgres
- If package.json has "dependencies: {redis: ^4.0.0}", extract: application DEPENDS_ON redis
- If docker-compose has service definitions, extract: service DEPLOYED_WITH database
        """,
        ),
    ]


def reflexion(context: dict[str, Any]) -> list[Message]:
    sys_prompt = """You are an AI assistant that determines which GitHub repository relationships have not been extracted from the given context"""

    user_prompt = f"""
<PREVIOUS MESSAGES>
{json.dumps([ep for ep in context['previous_episodes']], indent=2)}
</PREVIOUS MESSAGES>
<CURRENT MESSAGE>
{context['episode_content']}
</CURRENT MESSAGE>

<EXTRACTED ENTITIES>
{context['nodes']}
</EXTRACTED ENTITIES>

<EXTRACTED FACTS>
{context['extracted_facts']}
</EXTRACTED FACTS>

Given the above GitHub repository context, list of entities, and extracted relationships:
- Check for missing code dependencies (DEPENDS_ON, IMPORTS)
- Check for missing service connections (CONNECTS_TO, QUERIES, CACHES_IN)
- Check for missing deployment relationships (DEPLOYED_WITH, RUNS_IN)
- Check for missing repository structure relationships (CONTAINS, PART_OF)

Focus on technical relationships that would be found in code, config files, or documentation.
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt),
    ]


def extract_attributes(context: dict[str, Any]) -> list[Message]:
    return [
        Message(
            role='system',
            content='You are a helpful assistant that extracts relationship properties from GitHub repository content.',
        ),
        Message(
            role='user',
            content=f"""
<MESSAGE>
{json.dumps(context['episode_content'], indent=2)}
</MESSAGE>
<REFERENCE TIME>
{context['reference_time']}
</REFERENCE TIME>

Given the above GitHub repository content and the following relationship, update any attributes based on:
- Version information from package files
- Configuration details from config files
- Deployment information from docker-compose or CI/CD files
- Connection parameters from code

<FACT>
{context['fact']}
</FACT>
        """,
        ),
    ]


versions: Versions = {
    'edge': edge,
    'reflexion': reflexion,
    'extract_attributes': extract_attributes,
}