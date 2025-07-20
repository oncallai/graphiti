"""
Monitoring resources specific prompts for edge extraction
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
            content='You are an expert at extracting monitoring and observability relationships. '
            'Focus on metric collection, alerting chains, and observability flows.',
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
Extract relationships between monitoring systems, monitored services, and observability components.

# MONITORING-SPECIFIC RELATIONSHIPS TO EXTRACT:

1. **Monitoring Coverage**:
   - MONITORS: Monitoring relationships (prometheus MONITORS application)
   - COLLECTS_METRICS_FROM: Metric collection (prometheus COLLECTS_METRICS_FROM exporter)
   - SCRAPES: Scraping relationships (prometheus SCRAPES endpoint)
   - TRACKS: Tracking relationships (apm TRACKS transaction)
   - OBSERVES: Observation relationships (datadog OBSERVES infrastructure)

2. **Data Flow**:
   - EXPORTS_TO: Metric export (exporter EXPORTS_TO prometheus)
   - AGGREGATES_FROM: Aggregation (grafana AGGREGATES_FROM prometheus)
   - QUERIES: Query relationships (dashboard QUERIES datasource)
   - FORWARDS_TO: Log forwarding (fluentd FORWARDS_TO elasticsearch)
   - STORES_METRICS_IN: Storage relationships (prometheus STORES_METRICS_IN tsdb)

3. **Alerting Chain**:
   - ALERTS_ON: Alert conditions (alertmanager ALERTS_ON service)
   - NOTIFIES: Notification flow (alert NOTIFIES pagerduty)
   - ESCALATES_TO: Escalation (primary_oncall ESCALATES_TO secondary_oncall)
   - TRIGGERS: Alert triggers (metric_threshold TRIGGERS alert)
   - PAGES: Paging relationships (alert PAGES oncall_team)

4. **Visualization**:
   - DISPLAYS: Display relationships (dashboard DISPLAYS metrics)
   - VISUALIZES: Visualization (grafana VISUALIZES prometheus_data)
   - SHOWS_STATUS_OF: Status display (dashboard SHOWS_STATUS_OF service)
   - RENDERS: Rendering relationships (panel RENDERS timeseries)

5. **Trace & Log Relationships**:
   - TRACES: Distributed tracing (jaeger TRACES microservice)
   - CORRELATES_WITH: Correlation (trace CORRELATES_WITH logs)
   - LOGS_TO: Logging relationships (application LOGS_TO elasticsearch)
   - INDEXES: Index relationships (elasticsearch INDEXES application_logs)

6. **Integration & Federation**:
   - FEDERATES_WITH: Federation (prometheus FEDERATES_WITH central_prometheus)
   - REMOTE_WRITES_TO: Remote storage (prometheus REMOTE_WRITES_TO cortex)
   - PULLS_METRICS_FROM: Pull-based collection (victoria_metrics PULLS_METRICS_FROM prometheus)
   - SUBSCRIBES_TO: Event subscription (splunk SUBSCRIBES_TO event_stream)

# EXTRACTION RULES FOR MONITORING:

1. Focus on observability flows and data pipelines
2. Include both push and pull-based relationships
3. Capture alerting and escalation chains
4. Extract from:
   - Prometheus configurations
   - Grafana dashboard JSON
   - AlertManager rules
   - APM configurations
   - Log pipeline configs

{context['custom_prompt']}

# EXAMPLES:
- "Prometheus scrapes app metrics" → prometheus SCRAPES application
- "Grafana dashboard queries Prometheus" → dashboard QUERIES prometheus
- "Alert notifies Slack channel" → alert NOTIFIES slack_channel
- "Jaeger traces user-service" → jaeger TRACES user_service
- "Logs forwarded to Elasticsearch" → fluentd FORWARDS_TO elasticsearch
        """,
        ),
    ]


def reflexion(context: dict[str, Any]) -> list[Message]:
    sys_prompt = """You are an AI assistant that determines which monitoring relationships have not been extracted from the given context"""

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

Given the above monitoring context, check for missing:
- Monitoring coverage relationships (MONITORS, COLLECTS_METRICS_FROM, SCRAPES)
- Data flow relationships (EXPORTS_TO, QUERIES, FORWARDS_TO)
- Alerting relationships (ALERTS_ON, NOTIFIES, ESCALATES_TO)
- Visualization relationships (DISPLAYS, VISUALIZES)
- Trace/log relationships (TRACES, LOGS_TO)

Focus on observability pipelines and monitoring topology.
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt),
    ]


def extract_attributes(context: dict[str, Any]) -> list[Message]:
    return [
        Message(
            role='system',
            content='You are a helpful assistant that extracts monitoring relationship properties.',
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

Given the above monitoring content and the following relationship, update attributes based on:
- Metric names or patterns
- Scrape intervals or frequencies
- Alert thresholds or conditions
- Dashboard panel configurations
- Log index patterns
- Retention policies

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