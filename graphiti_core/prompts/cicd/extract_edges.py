"""
CI/CD-specific prompts for edge extraction
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
            content='You are an expert at extracting CI/CD pipeline relationships. '
            'Focus on build dependencies, deployment workflows, and pipeline stage relationships.',
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
Extract relationships between CI/CD pipeline entities.

# CI/CD-SPECIFIC RELATIONSHIPS TO EXTRACT:

1. **Pipeline & Stage Relationships**:
   - TRIGGERED_BY: Pipeline triggers (pipeline TRIGGERED_BY code_push)
   - CONTAINS: Stage containment (pipeline CONTAINS build_stage)
   - PRECEDES: Stage ordering (build_stage PRECEDES test_stage)
   - DEPENDS_ON: Stage dependencies (deploy_stage DEPENDS_ON test_stage)
   - PARALLEL_TO: Parallel execution (unit_tests PARALLEL_TO integration_tests)
   - CONDITIONAL_ON: Conditional execution (deploy_stage CONDITIONAL_ON test_success)

2. **Build & Test Relationships**:
   - BUILDS: Build processes (job BUILDS application)
   - TESTS: Testing relationships (job TESTS code)
   - VALIDATES: Validation processes (job VALIDATES configuration)
   - SCANS: Security scanning (job SCANS vulnerabilities)
   - ANALYZES: Code analysis (job ANALYZES quality)
   - GENERATES: Artifact generation (job GENERATES docker_image)

3. **Deployment & Environment Relationships**:
   - DEPLOYS_TO: Deployment targets (pipeline DEPLOYS_TO production)
   - PROMOTES_TO: Environment promotion (staging PROMOTES_TO production)
   - ROLLS_BACK_TO: Rollback relationships (production ROLLS_BACK_TO previous_version)
   - BLUE_GREEN_TO: Blue-green deployment (blue_environment BLUE_GREEN_TO green_environment)
   - CANARY_TO: Canary deployment (canary_environment CANARY_TO production)
   - SCALES_IN: Scaling relationships (application SCALES_IN kubernetes)

4. **Artifact & Package Relationships**:
   - PUBLISHES_TO: Artifact publishing (build PUBLISHES_TO artifact_repository)
   - PULLS_FROM: Artifact retrieval (deploy PULLS_FROM container_registry)
   - PUSHES_TO: Image pushing (build PUSHES_TO docker_registry)
   - STORES_IN: Storage relationships (artifacts STORES_IN nexus)
   - VERSIONS_WITH: Versioning (artifact VERSIONS_WITH git_tag)
   - SIGNS_WITH: Code signing (artifact SIGNS_WITH certificate)

5. **Infrastructure & Configuration Relationships**:
   - PROVISIONED_BY: Infrastructure provisioning (environment PROVISIONED_BY terraform)
   - CONFIGURED_BY: Configuration management (server CONFIGURED_BY ansible)
   - SECURED_BY: Security tools (pipeline SECURED_BY vault)
   - MONITORED_BY: Monitoring relationships (deployment MONITORED_BY prometheus)
   - LOGGED_TO: Logging (application LOGGED_TO cloudwatch)
   - ALERTED_BY: Alerting (service ALERTED_BY pagerduty)

6. **Tool & Platform Relationships**:
   - RUNS_ON: Execution environment (job RUNS_ON jenkins_agent)
   - EXECUTED_BY: Tool execution (pipeline EXECUTED_BY github_actions)
   - MANAGED_BY: Management relationships (build MANAGED_BY teamcity)
   - INTEGRATED_WITH: Tool integration (jenkins INTEGRATED_WITH sonarqube)
   - NOTIFIED_BY: Notifications (pipeline NOTIFIED_BY slack)
   - REPORTED_TO: Reporting (test_results REPORTED_TO jira)

7. **Quality & Compliance Relationships**:
   - COMPLIANT_WITH: Compliance checking (code COMPLIANT_WITH security_policy)
   - AUDITED_BY: Audit processes (deployment AUDITED_BY compliance_tool)
   - APPROVED_BY: Approval workflows (deploy APPROVED_BY security_team)
   - VALIDATED_BY: Validation processes (config VALIDATED_BY linting_tool)
   - REVIEWED_BY: Code review (change REVIEWED_BY senior_developer)
   - CERTIFIED_BY: Certification (release CERTIFIED_BY qa_team)

# EXTRACTION RULES FOR CI/CD:

1. Focus on pipeline stage dependencies and ordering
2. Include build and deployment workflow relationships
3. Capture artifact and package management relationships
4. Extract from:
   - Pipeline configuration files
   - Build and deployment logs
   - CI/CD platform configurations
   - Infrastructure as Code files
   - Monitoring and alerting configurations

{context['custom_prompt']}

# EXAMPLES:
- "build stage precedes test stage" → build_stage PRECEDES test_stage
- "pipeline deploys to production" → pipeline DEPLOYS_TO production
- "build publishes to artifact repository" → build PUBLISHES_TO artifact_repository
- "deployment monitored by Prometheus" → deployment MONITORED_BY prometheus
- "pipeline notified by Slack" → pipeline NOTIFIED_BY slack
        """,
        ),
    ]


def reflexion(context: dict[str, Any]) -> list[Message]:
    sys_prompt = """You are an AI assistant that determines which CI/CD relationships have not been extracted from the given context"""

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
- Pipeline and stage relationships (TRIGGERED_BY, CONTAINS, PRECEDES, DEPENDS_ON)
- Build and test relationships (BUILDS, TESTS, VALIDATES, SCANS)
- Deployment and environment relationships (DEPLOYS_TO, PROMOTES_TO, ROLLS_BACK_TO)
- Artifact and package relationships (PUBLISHES_TO, PULLS_FROM, STORES_IN)
- Infrastructure and configuration relationships (PROVISIONED_BY, CONFIGURED_BY, MONITORED_BY)
- Tool and platform relationships (RUNS_ON, EXECUTED_BY, INTEGRATED_WITH)
- Quality and compliance relationships (COMPLIANT_WITH, AUDITED_BY, APPROVED_BY)

Focus on CI/CD pipeline workflow and deployment relationships.
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
- Pipeline stage configurations and conditions
- Build and deployment environment settings
- Artifact versioning and storage information
- Infrastructure provisioning details
- Monitoring and alerting configurations
- Security and compliance requirements
- Tool integration and notification settings

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