"""
Application-centric prompts for cross-domain edge extraction
This prompt creates relationships between applications and their resources across all domains
to enable comprehensive system understanding and troubleshooting.
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
            content='You are an expert at creating cross-domain application relationships. '
            'Your goal is to connect applications to all their resources across GitHub, Cloud, CI/CD, and Observability domains '
            'to enable comprehensive system understanding and troubleshooting.',
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
Create cross-domain relationships that connect applications to all their resources across domains.
This enables comprehensive system understanding and troubleshooting.

# APPLICATION-CENTRIC RELATIONSHIPS TO EXTRACT:

## 1. **Application Ownership & Deployment**:
   - DEPLOYED_AS: Application deployment relationships (e-commerce-platform DEPLOYED_AS production-service)
   - RUNS_ON: Execution platform (application RUNS_ON kubernetes-cluster)
   - HOSTED_IN: Hosting relationships (web-app HOSTED_IN aws-ec2)
   - CONTAINED_IN: Container relationships (service CONTAINED_IN docker-container)

## 2. **GitHub & Development Connections**:
   - SOURCE_CODE_IN: Repository relationships (application SOURCE_CODE_IN github-repo)
   - BUILT_FROM: Build source (service BUILT_FROM main-branch)
   - VERSIONED_IN: Version control (app VERSIONED_IN release-tag)
   - DEVELOPED_BY: Development team (application DEVELOPED_BY backend-team)

## 3. **Cloud Infrastructure Dependencies**:
   - USES_DATABASE: Database connections (app USES_DATABASE postgres-instance)
   - CACHES_IN: Caching relationships (service CACHES_IN redis-cluster)
   - STORES_DATA_IN: Storage relationships (app STORES_DATA_IN s3-bucket)
   - CONNECTS_TO: Service connections (api CONNECTS_TO external-service)
   - LOAD_BALANCED_BY: Load balancing (app LOAD_BALANCED_BY alb)

## 4. **CI/CD Pipeline Integration**:
   - DEPLOYED_VIA: Deployment pipeline (app DEPLOYED_VIA github-actions)
   - TESTED_BY: Testing relationships (service TESTED_BY automated-tests)
   - MONITORED_BY: Pipeline monitoring (deploy MONITORED_BY jenkins)
   - ARTIFACT_FROM: Build artifacts (deployment ARTIFACT_FROM docker-image)

## 5. **Observability & Monitoring**:
   - MONITORED_BY: Monitoring systems (app MONITORED_BY prometheus)
   - LOGS_TO: Logging relationships (service LOGS_TO elasticsearch)
   - TRACED_BY: Distributed tracing (app TRACED_BY jaeger)
   - ALERTED_ON: Alerting relationships (service ALERTED_ON pagerduty)
   - METRICS_COLLECTED_BY: Metrics collection (app METRICS_COLLECTED_BY datadog)

## 6. **Cross-Domain Dependencies**:
   - DEPENDS_ON: Service dependencies (frontend DEPENDS_ON backend-api)
   - COMMUNICATES_WITH: Inter-service communication (auth-service COMMUNICATES_WITH user-db)
   - SHARES_DATA_WITH: Data sharing (analytics SHARES_DATA_WITH reporting-dashboard)
   - INTEGRATES_WITH: External integrations (payment-service INTEGRATES_WITH stripe)

# EXTRACTION RULES FOR APPLICATION-CENTRIC RELATIONSHIPS:

1. **Identify the Main Application**: Look for the primary application name across all domains
2. **Connect All Resources**: Create relationships from the application to all its resources
3. **Cross-Domain Mapping**: Connect resources that belong to the same application across domains
4. **Enable Troubleshooting**: Create relationships that help understand system dependencies
5. **Extract from**:
   - Application names in GitHub repositories
   - Service names in cloud configurations
   - Pipeline names in CI/CD configurations
   - Service names in monitoring configurations
   - Common naming patterns across domains

# EXAMPLES:
- "E-Commerce Platform" → "ecommerce-platform SOURCE_CODE_IN github-repo"
- "E-Commerce Platform" → "ecommerce-platform USES_DATABASE postgres-db"
- "E-Commerce Platform" → "ecommerce-platform DEPLOYED_VIA github-actions"
- "E-Commerce Platform" → "ecommerce-platform MONITORED_BY prometheus"
- "User Service" → "user-service CONNECTS_TO auth-service"
- "Analytics Platform" → "analytics-platform SHARES_DATA_WITH reporting-dashboard"

{context['custom_prompt']}
        """,
        ),
    ]


def reflexion(context: dict[str, Any]) -> list[Message]:
    sys_prompt = """You are an AI assistant that determines which cross-domain application relationships have not been extracted from the given context"""

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

Given the above cross-domain context, check for missing:
- Application ownership relationships (DEPLOYED_AS, RUNS_ON, HOSTED_IN)
- GitHub connections (SOURCE_CODE_IN, BUILT_FROM, VERSIONED_IN)
- Cloud infrastructure dependencies (USES_DATABASE, CACHES_IN, STORES_DATA_IN)
- CI/CD pipeline integration (DEPLOYED_VIA, TESTED_BY, MONITORED_BY)
- Observability relationships (MONITORED_BY, LOGS_TO, TRACED_BY)
- Cross-domain dependencies (DEPENDS_ON, COMMUNICATES_WITH, INTEGRATES_WITH)

Focus on creating a complete picture of how applications connect to all their resources across domains.
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt),
    ]


def extract_attributes(context: dict[str, Any]) -> list[Message]:
    return [
        Message(
            role='system',
            content='You are a helpful assistant that extracts cross-domain relationship properties.',
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

Given the above cross-domain content and the following relationship, update attributes based on:
- Application version information
- Deployment environment details
- Cross-domain configuration parameters
- Integration settings and endpoints
- Monitoring and alerting configurations
- Service dependency information

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