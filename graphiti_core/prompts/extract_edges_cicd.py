"""
CI/CD resources specific prompts for edge extraction
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
            content='You are an expert at extracting CI/CD workflow relationships. '
            'Focus on build dependencies, deployment flows, and automation relationships.',
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
Extract relationships between CI/CD pipeline entities and automation tools.

# CI/CD-SPECIFIC RELATIONSHIPS TO EXTRACT:

1. **Pipeline Flow**:
   - TRIGGERS: Trigger relationships (commit TRIGGERS build_pipeline)
   - FOLLOWS: Sequential stages (build_stage FOLLOWS test_stage)
   - DEPENDS_ON_STAGE: Stage dependencies (deploy DEPENDS_ON_STAGE build)
   - PROMOTES_TO: Environment promotion (dev_deploy PROMOTES_TO staging_deploy)
   - PARALLEL_WITH: Parallel execution (test_job PARALLEL_WITH lint_job)

2. **Build Relationships**:
   - BUILDS: Build production (pipeline BUILDS docker_image)
   - COMPILES: Compilation (build_job COMPILES application)
   - PACKAGES: Packaging (pipeline PACKAGES artifact)
   - PUBLISHES_TO: Artifact publishing (build PUBLISHES_TO artifactory)
   - PULLS_FROM: Dependency retrieval (build PULLS_FROM npm_registry)

3. **Deployment Relationships**:
   - DEPLOYS_TO: Deployment targets (pipeline DEPLOYS_TO kubernetes)
   - UPDATES: Update relationships (deploy_job UPDATES service)
   - ROLLS_BACK_TO: Rollback relationships (pipeline ROLLS_BACK_TO previous_version)
   - CONFIGURES: Configuration deployment (pipeline CONFIGURES application)
   - PROVISIONS: Infrastructure provisioning (terraform_job PROVISIONS resources)

4. **Testing & Quality**:
   - TESTS: Test execution (test_job TESTS application)
   - SCANS: Security/quality scanning (sonarqube SCANS codebase)
   - VALIDATES: Validation (smoke_test VALIDATES deployment)
   - GATES: Quality gates (quality_gate GATES deployment)

5. **Tool Integration**:
   - EXECUTED_BY: Execution relationships (job EXECUTED_BY jenkins)
   - RUNS_ON_AGENT: Agent relationships (build RUNS_ON_AGENT build_agent)
   - NOTIFIES: Notification (pipeline NOTIFIES slack_channel)
   - REPORTS_TO: Reporting (pipeline REPORTS_TO dashboard)
   - INTEGRATES_WITH: Tool integration (jenkins INTEGRATES_WITH github)

6. **Artifact Flow**:
   - PRODUCES: Artifact production (build PRODUCES war_file)
   - CONSUMES: Artifact consumption (deploy CONSUMES docker_image)
   - STORES_IN: Artifact storage (pipeline STORES_IN nexus)
   - FETCHES_FROM: Artifact retrieval (deploy FETCHES_FROM registry)

# EXTRACTION RULES FOR CI/CD:

1. Focus on workflow and automation relationships
2. Capture both build-time and deploy-time relationships
3. Include environment-specific relationships
4. Extract from:
   - Pipeline definitions (Jenkinsfile, .gitlab-ci.yml)
   - Workflow files (GitHub Actions)
   - Deployment scripts
   - Configuration management files
   - Tool configurations

{context['custom_prompt']}

# EXAMPLES:
- "Build stage triggers deploy stage" → build_stage TRIGGERS deploy_stage
- "Jenkins builds Docker image" → jenkins_pipeline BUILDS docker_image
- "Deploy to production after staging" → staging_deploy PROMOTES_TO prod_deploy
- "Pipeline publishes to Artifactory" → pipeline PUBLISHES_TO artifactory
        """,
        ),
    ]


def reflexion(context: dict[str, Any]) -> list[Message]:
    sys_prompt = """You are an AI assistant that determines which CI/CD workflow relationships have not been extracted from the given context"""

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

Given the above CI/CD context, check for missing:
- Pipeline flow relationships (TRIGGERS, FOLLOWS, DEPENDS_ON_STAGE)
- Build relationships (BUILDS, PACKAGES, PUBLISHES_TO)
- Deployment relationships (DEPLOYS_TO, UPDATES, PROVISIONS)
- Tool integration relationships (EXECUTED_BY, INTEGRATES_WITH)
- Artifact flow relationships (PRODUCES, CONSUMES, STORES_IN)

Focus on automation workflows and tool integrations.
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt),
    ]


def extract_attributes(context: dict[str, Any]) -> list[Message]:
    return [
        Message(
            role='system',
            content='You are a helpful assistant that extracts CI/CD relationship properties.',
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

Given the above CI/CD content and the following relationship, update attributes based on:
- Pipeline stage information
- Build numbers or versions
- Environment details
- Trigger conditions
- Artifact versions
- Tool configurations

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