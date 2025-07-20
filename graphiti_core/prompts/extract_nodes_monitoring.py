"""
Monitoring resources specific prompts for node extraction
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
    sys_prompt = """You are an AI assistant specialized in extracting monitoring and observability entities.
    Your primary focus is identifying monitoring systems, metrics, alerts, and observed services."""

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

You are analyzing monitoring resources. Extract entities focusing on:

## PRIMARY EXTRACTION TARGETS:

1. **Monitoring Systems**:
   - Prometheus instances
   - Grafana instances
   - Elasticsearch clusters
   - Datadog agents
   - New Relic agents
   - CloudWatch namespaces

2. **Monitored Services**:
   - Application names being monitored
   - Service names with metrics
   - Database instances being tracked
   - Infrastructure components under observation
   - API endpoints being monitored

3. **Observability Components**:
   - Log aggregators (ELK, Splunk, Fluentd)
   - APM tools (AppDynamics, Dynatrace)
   - Tracing systems (Jaeger, Zipkin)
   - Metric exporters
   - Service mesh observability (Istio, Linkerd)

4. **Alerting Infrastructure**:
   - Alert manager instances
   - PagerDuty services
   - Slack channels for alerts
   - Email distribution lists
   - Notification webhooks

## MONITORING-SPECIFIC RULES:
- Extract both monitoring tools AND monitored services
- Include dashboard names when they represent key services
- Capture alert rule names for critical alerts
- Focus on production monitoring entities
- Include data retention systems (long-term storage)

## NAMING CONVENTIONS:
- Prometheus: job names and instance labels
- Grafana: dashboard names and data source names
- ELK: index patterns and Kibana space names
- Use service names as they appear in metrics

## DO NOT EXTRACT:
- Individual metric names
- Threshold values or SLO percentages
- Temporary debugging configurations
- Test or development monitoring setups

{context['custom_prompt']}
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt),
    ]


def extract_json(context: dict[str, Any]) -> list[Message]:
    sys_prompt = """You are an AI assistant specialized in extracting monitoring entities from JSON configurations.
    Focus on monitoring configs, dashboard definitions, and alert rules."""

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

You are analyzing JSON from monitoring configurations. This could be:
- Prometheus configuration files
- Grafana dashboard JSON
- AlertManager config
- Datadog monitors
- CloudWatch alarms
- Elasticsearch mappings

## EXTRACTION FOCUS FOR MONITORING JSON:

1. **From Prometheus Config**:
   - Job names from scrape_configs
   - Target service names
   - Service discovery endpoints
   - Remote storage endpoints

2. **From Grafana Dashboards**:
   - Dashboard titles
   - Data source names
   - Panel titles representing key services
   - Variable definitions for service names

3. **From Alert Configurations**:
   - Alert rule names
   - Target services in alerts
   - Notification channel names
   - Escalation policy names

## MONITORING JSON SPECIFIC RULES:
- Extract service names from job labels
- Include monitoring namespace/tenant names
- Focus on service identifiers, not metric names
- Extract integration endpoints (webhook URLs → service names)
- Capture monitoring cluster/instance names

{context['custom_prompt']}
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt),
    ]


def extract_text(context: dict[str, Any]) -> list[Message]:
    sys_prompt = """You are an AI assistant specialized in extracting monitoring entities from documentation.
    Focus on monitoring guides, runbooks, and observability documentation."""

    user_prompt = f"""
<TEXT>
{context['episode_content']}
</TEXT>
<ENTITY TYPES>
{context['entity_types']}
</ENTITY TYPES>

You are analyzing text about monitoring and observability. This could be:
- Monitoring architecture documentation
- Alert runbooks
- Dashboard descriptions
- Incident response procedures
- Observability strategy documents

## EXTRACTION FOCUS FOR MONITORING TEXT:

1. **Monitoring Infrastructure**:
   - Monitoring system names and instances
   - Time series database names
   - Log storage systems
   - Visualization tool names

2. **Monitored Services**:
   - Critical services under monitoring
   - Service dependencies being tracked
   - Database instances being monitored
   - Infrastructure components with alerts

3. **Observability Components**:
   - Tracing system names
   - Log pipeline components
   - Metrics aggregation services
   - Synthetic monitoring targets

## MONITORING TEXT SPECIFIC RULES:
- Extract both tools and monitored services
- Include SLO/SLA tracked services
- Capture incident management tools
- Focus on production monitoring setup
- Extract on-call rotation names/services

## NAMING STANDARDS:
- Use exact service names as they appear in monitoring
- Include environment context for monitored services
- Maintain monitoring tool canonical names
- Use full dashboard/alert names when significant

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