"""
GitHub-specific prompts for edge extraction
"""

import json
from typing import Any, Protocol, TypedDict
from pydantic import BaseModel, Field
from ..models import Message, PromptFunction, PromptVersion


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
            content='You are an expert at extracting relationships between GitHub repositories and development components. '
            'Focus on code dependencies, service connections, and development workflow relationships.',
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
Extract relationships between GitHub repositories and development entities.

# GITHUB-SPECIFIC RELATIONSHIPS TO EXTRACT:

1. **Repository & Code Dependencies**:
   - DEPENDS_ON: Package/library dependencies (app DEPENDS_ON library)
   - IMPORTS: Module imports (service IMPORTS module)
   - USES_FRAMEWORK: Application uses framework (app USES_FRAMEWORK express)
   - REQUIRES: Build/runtime requirements (service REQUIRES nodejs)
   - CONTAINS: Repository contains services (repo CONTAINS microservice)
   - PART_OF: Component hierarchy (module PART_OF application)

2. **Service & API Relationships**:
   - CONNECTS_TO: Service connections (api CONNECTS_TO database)
   - QUERIES: Database queries (service QUERIES postgres)
   - CACHES_IN: Caching relationships (api CACHES_IN redis)
   - PUBLISHES_TO: Message publishing (service PUBLISHES_TO rabbitmq)
   - CONSUMES_FROM: Message consumption (worker CONSUMES_FROM kafka)
   - EXPOSES: API exposure (service EXPOSES rest_api)

3. **GitHub Workflow Relationships**:
   - TRIGGERED_BY: Workflow triggers (workflow TRIGGERED_BY push)
   - DEPLOYS_TO: Deployment targets (workflow DEPLOYS_TO production)
   - TESTS: Testing relationships (workflow TESTS application)
   - BUILDS: Build processes (workflow BUILDS docker_image)
   - NOTIFIES: Notification relationships (workflow NOTIFIES slack)
   - RUNS_ON: Execution environment (workflow RUNS_ON self_hosted_runner)

4. **Branch & Version Relationships**:
   - MERGES_TO: Branch merging (feature_branch MERGES_TO main)
   - CREATES: Branch creation (main CREATES release_branch)
   - TAGS: Version tagging (commit TAGS v1.0.0)
   - RELEASES: Release relationships (tag RELEASES version)
   - FORKS: Repository forking (repo FORKS original_repo)
   - CLONES: Repository cloning (local_repo CLONES remote_repo)

5. **Development Tool Relationships**:
   - MONITORED_BY: Monitoring tools (app MONITORED_BY sentry)
   - TESTED_BY: Testing frameworks (code TESTED_BY jest)
   - LINTED_BY: Code quality (code LINTED_BY eslint)
   - DOCUMENTED_BY: Documentation (api DOCUMENTED_BY swagger)
   - BACKED_UP_BY: Backup systems (repo BACKED_UP_BY github)
   - SECURED_BY: Security tools (repo SECURED_BY dependabot)

6. **Environment & Deployment**:
   - DEPLOYED_TO: Environment deployment (app DEPLOYED_TO staging)
   - RUNS_IN: Runtime environment (service RUNS_IN docker)
   - HOSTED_ON: Hosting platform (app HOSTED_ON heroku)
   - SCALED_BY: Scaling relationships (app SCALED_BY kubernetes)
   - LOAD_BALANCED_BY: Load balancing (api LOAD_BALANCED_BY nginx)
   - PROXIED_BY: Proxy relationships (service PROXIED_BY cloudflare)

7. **External Service Integration**:
   - AUTHENTICATED_BY: Authentication (app AUTHENTICATED_BY oauth)
   - AUTHORIZED_BY: Authorization (service AUTHORIZED_BY jwt)
   - ENCRYPTED_BY: Encryption (data ENCRYPTED_BY ssl)
   - STORED_IN: Storage relationships (files STORED_IN s3)
   - LOGGED_TO: Logging (app LOGGED_TO cloudwatch)
   - METRICED_BY: Metrics collection (service METRICED_BY prometheus)

# EXTRACTION RULES FOR GITHUB:

1. Focus on code dependencies and service relationships
2. Include GitHub-specific workflow relationships
3. Capture both direct and indirect dependencies
4. Extract from:
   - package.json and dependency files
   - GitHub Actions workflows
   - Configuration files
   - Documentation and README files
   - Code comments and imports

{context['custom_prompt']}

# EXAMPLES:
- "app depends on express framework" → app DEPENDS_ON express
- "API connects to PostgreSQL database" → api CONNECTS_TO postgresql
- "workflow deploys to production" → workflow DEPLOYS_TO production
- "feature branch merges to main" → feature_branch MERGES_TO main
- "app monitored by Sentry" → app MONITORED_BY sentry
        """,
        ),
    ]


def reflexion(context: dict[str, Any]) -> list[Message]:
    sys_prompt = """You are an AI assistant that determines which GitHub relationships have not been extracted from the given context"""

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

Given the above GitHub context, check for missing:
- Repository and code dependencies (DEPENDS_ON, IMPORTS, USES_FRAMEWORK)
- Service and API relationships (CONNECTS_TO, QUERIES, EXPOSES)
- GitHub workflow relationships (TRIGGERED_BY, DEPLOYS_TO, RUNS_ON)
- Branch and version relationships (MERGES_TO, TAGS, RELEASES)
- Development tool relationships (MONITORED_BY, TESTED_BY, LINTED_BY)
- Environment and deployment relationships (DEPLOYED_TO, RUNS_IN, HOSTED_ON)
- External service integration (AUTHENTICATED_BY, STORED_IN, LOGGED_TO)

Focus on development workflow and code dependency relationships.
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt),
    ]


def extract_attributes(context: dict[str, Any]) -> list[Message]:
    return [
        Message(
            role='system',
            content='You are a helpful assistant that extracts GitHub relationship properties.',
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

Given the above GitHub content and the following relationship, update attributes based on:
- Repository and branch information
- Dependency versions and constraints
- Workflow trigger conditions
- Environment configurations
- Service endpoints and URLs
- Authentication and security settings
- Deployment and scaling configurations

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