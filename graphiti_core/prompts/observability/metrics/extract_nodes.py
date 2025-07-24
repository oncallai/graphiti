"""
Observability Metrics-specific prompts for node extraction
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
    sys_prompt = """You are an AI assistant specialized in extracting metrics and monitoring entities.
    Your primary focus is identifying metrics collection systems, monitoring tools, and performance measurement platforms."""

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

You are analyzing metrics and monitoring systems. Extract entities focusing on:

## PRIMARY EXTRACTION TARGETS:

1. **Metrics Collection Platforms**:
   - Prometheus and Prometheus-compatible systems
   - Grafana and visualization platforms
   - Datadog monitoring and analytics
   - New Relic APM and monitoring
   - AWS CloudWatch Metrics
   - Azure Monitor Metrics
   - Google Cloud Monitoring
   - InfluxDB and time-series databases
   - OpenTelemetry collectors
   - StatsD and metrics aggregators

2. **Application Performance Monitoring (APM)**:
   - APM tools and platforms
   - Application performance dashboards
   - Service performance monitors
   - Database performance monitoring
   - API performance tracking
   - User experience monitoring
   - Real user monitoring (RUM)
   - Synthetic monitoring tools

3. **Infrastructure Monitoring**:
   - Server and host monitoring
   - Container and Kubernetes monitoring
   - Network monitoring systems
   - Storage monitoring tools
   - Cloud resource monitoring
   - Virtual machine monitoring
   - Load balancer monitoring
   - Database monitoring systems

4. **Business Metrics & KPIs**:
   - Business intelligence dashboards
   - Key performance indicators
   - Revenue and financial metrics
   - User engagement metrics
   - Conversion tracking systems
   - Customer satisfaction metrics
   - Operational efficiency metrics
   - SLA and SLO monitoring

5. **Alerting & Notification Systems**:
   - Alert management platforms
   - Notification systems (PagerDuty, Slack, etc.)
   - Escalation management tools
   - Incident response systems
   - On-call management platforms
   - Alert routing and filtering
   - Alert correlation engines
   - Status page systems

6. **Metrics Storage & Processing**:
   - Time-series databases
   - Metrics aggregation systems
   - Data retention and archival
   - Metrics compression and optimization
   - Metrics federation and sharing
   - Metrics backup and recovery
   - Metrics security and access control

7. **Custom Metrics & Instrumentation**:
   - Custom metric collectors
   - Application instrumentation
   - Business logic metrics
   - Custom dashboards and reports
   - Metric exporters and integrations
   - Custom alerting rules
   - Metric transformation tools

## METRICS-SPECIFIC RULES:
- Extract metrics platform and system names
- Include monitoring tool and dashboard names
- Capture metric collection agent names
- Extract from monitoring configurations and manifests
- Include environment names (prod, staging, dev)
- Capture alerting and notification system names

## METRICS NAMING CONVENTIONS:
- Metrics platforms: use platform names (e.g., "prometheus-cluster", "grafana-instance")
- Monitoring tools: use tool names (e.g., "datadog-agent", "newrelic-apm")
- Dashboards: use dashboard names (e.g., "application-metrics-dashboard", "infrastructure-overview")
- Alerts: use alert rule names (e.g., "high-cpu-alert", "service-down-notification")

## DO NOT EXTRACT:
- Generic metric names without context
- Temporary metric files or artifacts
- Metric configuration values or parameters
- Personal information or credentials
- Generic monitoring terms without specific instances

{context['custom_prompt']}
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt),
    ]


def extract_json(context: dict[str, Any]) -> list[Message]:
    sys_prompt = """You are an AI assistant specialized in extracting metrics entities from JSON configurations.
    Focus on monitoring platform configurations, metric definitions, and alerting rule settings."""

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

You are analyzing JSON from metrics systems. This could be:
- Monitoring platform configurations
- Metric definitions and rules
- Alerting and notification settings
- Dashboard configurations
- Metrics collection agent settings
- APM tool configurations

## EXTRACTION FOCUS FOR METRICS JSON:

1. **From Monitoring Platform Configs**:
   - Platform names and cluster names
   - Metric and alert rule names
   - Dashboard and visualization names
   - Data source and target names
   - Integration and API endpoint names

2. **From Metric Definitions**:
   - Metric names and identifiers
   - Collection agent names
   - Data source and target names
   - Processing and transformation names
   - Storage and retention policy names

3. **From Alerting Configurations**:
   - Alert rule names and identifiers
   - Notification channel names
   - Escalation policy names
   - Incident management system names
   - Status page and communication names

4. **From Dashboard Configs**:
   - Dashboard and panel names
   - Visualization and chart names
   - Data source and query names
   - Filter and variable names
   - Export and sharing configuration names

## METRICS JSON SPECIFIC RULES:
- Extract monitoring platform names from appropriate fields
- Include metric collection agent and tool names
- Capture alerting and notification system names
- Focus on dashboard and visualization names
- Skip generic configuration values

{context['custom_prompt']}
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt),
    ]


def extract_text(context: dict[str, Any]) -> list[Message]:
    sys_prompt = """You are an AI assistant specialized in extracting metrics entities from documentation and reports.
    Focus on monitoring documentation, performance reports, and metrics system descriptions."""

    user_prompt = f"""
<TEXT>
{context['episode_content']}
</TEXT>
<ENTITY TYPES>
{context['entity_types']}
</ENTITY TYPES>

You are analyzing text about metrics systems. This could be:
- Monitoring system documentation
- Performance analysis reports
- Metrics configuration guides
- Monitoring dashboards
- Alerting and notification guides
- Metrics architecture documentation

## EXTRACTION FOCUS FOR METRICS TEXT:

1. **Monitoring Platform Information**:
   - Platform names and descriptions
   - Cluster and system names
   - Metric and alert rule names
   - Dashboard and visualization names
   - Integration and API names

2. **Metrics Collection Information**:
   - Collection agent and tool names
   - Data source and target names
   - Processing and transformation names
   - Storage and retention system names
   - Security and access control names

3. **Alerting and Notification Information**:
   - Alert rule and policy names
   - Notification channel names
   - Escalation and incident management names
   - Status page and communication names
   - Integration and webhook names

4. **Performance and APM Information**:
   - APM tool and platform names
   - Performance dashboard names
   - Service and application monitor names
   - Database and infrastructure monitor names
   - User experience monitoring names

## METRICS TEXT SPECIFIC RULES:
- Extract specific monitoring platform and tool names
- Include environment prefixes/suffixes (prod-, -staging)
- Capture metric collection agent and system names
- Focus on currently used monitoring tools and platforms
- Extract from configuration examples and reports

## METRICS NAMING STANDARDS:
- Use full monitoring platform names (e.g., "prometheus-production-cluster")
- Include environment context (e.g., "staging-application-metrics")
- Maintain original naming conventions
- Use canonical names for well-known monitoring platforms

{context['custom_prompt']}
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt),
    ]


def reflexion(context: dict[str, Any]) -> list[Message]:
    sys_prompt = """You are an AI assistant that determines which metrics entities have not been extracted from the given context"""

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

Given the above previous messages, current message, and list of extracted entities; determine if any metrics entities haven't been extracted.

Focus on missing:
- Metrics collection platforms and systems
- Application performance monitoring tools
- Infrastructure monitoring systems
- Business metrics and KPI tools
- Alerting and notification systems
- Metrics storage and processing tools
- Custom metrics and instrumentation tools
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt),
    ]


def classify_nodes(context: dict[str, Any]) -> list[Message]:
    sys_prompt = """You are an AI assistant that classifies metrics entity nodes given the context from which they were extracted"""

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
    
    Given the above conversation, extracted entities, and provided entity types and their descriptions, classify the extracted metrics entities.
    
    Guidelines:
    1. Each entity must have exactly one type
    2. Only use the provided ENTITY TYPES as types, do not use additional types to classify entities.
    3. If none of the provided entity types accurately classify an extracted node, the type should be set to None
    4. Consider metrics-specific context when classifying (e.g., monitoring platforms, APM tools, alerting systems)
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt),
    ]


def extract_attributes(context: dict[str, Any]) -> list[Message]:
    return [
        Message(
            role='system',
            content='You are a helpful assistant that extracts metrics entity properties from the provided text.',
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
        4. Consider metrics-specific attributes like collection intervals, retention policies, alert thresholds, etc.
        
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