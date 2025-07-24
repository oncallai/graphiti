"""
Observability Metrics-specific prompts for edge extraction
"""

import json
from typing import Any, Protocol, TypedDict
from pydantic import BaseModel, Field
from ...models import Message, PromptFunction, PromptVersion


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
            content='You are an expert at extracting metrics collection and monitoring relationships. '
            'Focus on metrics collection dependencies, monitoring system connections, and performance measurement relationships.',
        ),
        Message(
            role='user',
            content=f"""
<PREVIOUS_MESSAGES>
{json.dumps([ep for ep in context['previous_episodes']], indent=2)}
</PREVIOUS_MESSAGES>

<CURRENT_MESSAGE>
{context['episode_content']}
</CURRENT MESSAGE>

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
Extract relationships between metrics and monitoring entities.

# METRICS-SPECIFIC RELATIONSHIPS TO EXTRACT:

1. **Metrics Collection Relationships**:
   - COLLECTS_FROM: Metrics collection (prometheus COLLECTS_FROM application)
   - EXPORTS_TO: Metrics export (application EXPORTS_TO prometheus)
   - SCRAPES: Metrics scraping (prometheus SCRAPES node_exporter)
   - PUSHES_TO: Metrics pushing (application PUSHES_TO pushgateway)
   - PULLS_FROM: Metrics pulling (grafana PULLS_FROM prometheus)
   - STORES_IN: Metrics storage (prometheus STORES_IN tsdb)

2. **Monitoring & Alerting Relationships**:
   - MONITORS: Monitoring relationships (datadog MONITORS web_service)
   - ALERTS_ON: Alerting relationships (alert_rule ALERTS_ON high_cpu)
   - NOTIFIES: Notification relationships (alert NOTIFIES slack_channel)
   - ESCALATES_TO: Escalation relationships (alert ESCALATES_TO oncall_team)
   - CORRELATES_WITH: Alert correlation (cpu_alert CORRELATES_WITH memory_alert)
   - TRIGGERS: Alert triggering (threshold_breach TRIGGERS incident)

3. **Dashboard & Visualization Relationships**:
   - DISPLAYS: Dashboard display (dashboard DISPLAYS application_metrics)
   - QUERIES: Data querying (dashboard QUERIES prometheus)
   - VISUALIZES: Data visualization (chart VISUALIZES response_time)
   - FILTERS: Data filtering (dashboard FILTERS by_environment)
   - EXPORTS: Data export (dashboard EXPORTS to_pdf)
   - SHARES: Dashboard sharing (dashboard SHARES with_team)

4. **APM & Performance Relationships**:
   - TRACKS: Performance tracking (newrelic TRACKS api_performance)
   - MEASURES: Performance measurement (apm MEASURES response_time)
   - PROFILES: Performance profiling (tool PROFILES cpu_usage)
   - ANALYZES: Performance analysis (apm ANALYZES slow_queries)
   - OPTIMIZES: Performance optimization (tool OPTIMIZES database_queries)
   - BENCHMARKS: Performance benchmarking (tool BENCHMARKS throughput)

5. **Infrastructure Monitoring Relationships**:
   - WATCHES: Infrastructure watching (monitor WATCHES server_health)
   - CHECKS: Health checking (health_check CHECKS service_endpoint)
   - PROBES: Service probing (probe PROBES load_balancer)
   - PINGS: Network pinging (monitor PINGS network_device)
   - SCANS: Infrastructure scanning (scanner SCANS security_vulnerabilities)
   - AUDITS: Security auditing (auditor AUDITS access_logs)

6. **Data Flow & Processing Relationships**:
   - PROCESSES: Data processing (processor PROCESSES raw_metrics)
   - TRANSFORMS: Data transformation (transformer TRANSFORMS metric_format)
   - AGGREGATES: Data aggregation (aggregator AGGREGATES time_series)
   - FILTERS: Data filtering (filter FILTERS error_metrics)
   - ENRICHES: Data enrichment (enricher ENRICHES with_metadata)
   - VALIDATES: Data validation (validator VALIDATES metric_quality)

7. **Integration & API Relationships**:
   - INTEGRATES_WITH: Tool integration (prometheus INTEGRATES_WITH grafana)
   - CONNECTS_TO: API connections (dashboard CONNECTS_TO metrics_api)
   - WEBHOOKS_TO: Webhook relationships (alert WEBHOOKS_TO slack)
   - CALLS: API calling (collector CALLS metrics_endpoint)
   - SUBSCRIBES_TO: Event subscription (monitor SUBSCRIBES_TO health_events)
   - PUBLISHES_TO: Event publishing (service PUBLISHES_TO metrics_topic)

8. **Storage & Retention Relationships**:
   - BACKS_UP_TO: Data backup (metrics BACKS_UP_TO s3_bucket)
   - ARCHIVES_TO: Data archival (old_metrics ARCHIVES_TO cold_storage)
   - COMPRESSES: Data compression (compressor COMPRESSES time_series)
   - DEDUPLICATES: Data deduplication (deduplicator DEDUPLICATES metrics)
   - ENCRYPTS: Data encryption (encryptor ENCRYPTS sensitive_metrics)
   - RETENTION_POLICY: Retention management (policy RETENTION_POLICY metrics)

# EXTRACTION RULES FOR METRICS:

1. Focus on metrics collection and monitoring dependencies
2. Include alerting and notification workflow relationships
3. Capture dashboard and visualization data flow relationships
4. Extract from:
   - Monitoring platform configurations
   - Metrics collection agent settings
   - Alerting and notification configurations
   - Dashboard and visualization definitions
   - APM and performance monitoring setups

{context['custom_prompt']}

# EXAMPLES:
- "Prometheus collects metrics from application" → prometheus COLLECTS_FROM application
- "Alert rule monitors high CPU usage" → alert_rule ALERTS_ON high_cpu
- "Dashboard displays application metrics" → dashboard DISPLAYS application_metrics
- "New Relic tracks API performance" → newrelic TRACKS api_performance
- "Grafana queries Prometheus" → grafana QUERIES prometheus
        """,
        ),
    ]


def reflexion(context: dict[str, Any]) -> list[Message]:
    sys_prompt = """You are an AI assistant that determines which metrics relationships have not been extracted from the given context"""

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

Given the above metrics context, check for missing:
- Metrics collection relationships (COLLECTS_FROM, EXPORTS_TO, SCRAPES, PUSHES_TO)
- Monitoring and alerting relationships (MONITORS, ALERTS_ON, NOTIFIES, ESCALATES_TO)
- Dashboard and visualization relationships (DISPLAYS, QUERIES, VISUALIZES, FILTERS)
- APM and performance relationships (TRACKS, MEASURES, PROFILES, ANALYZES)
- Infrastructure monitoring relationships (WATCHES, CHECKS, PROBES, PINGS)
- Data flow and processing relationships (PROCESSES, TRANSFORMS, AGGREGATES, FILTERS)
- Integration and API relationships (INTEGRATES_WITH, CONNECTS_TO, WEBHOOKS_TO, CALLS)
- Storage and retention relationships (BACKS_UP_TO, ARCHIVES_TO, COMPRESSES, ENCRYPTS)

Focus on metrics collection and monitoring system relationships.
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt),
    ]


def extract_attributes(context: dict[str, Any]) -> list[Message]:
    return [
        Message(
            role='system',
            content='You are a helpful assistant that extracts metrics relationship properties.',
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

Given the above metrics content and the following relationship, update attributes based on:
- Metrics collection intervals and frequencies
- Alert thresholds and conditions
- Dashboard refresh rates and time ranges
- Data retention and archival policies
- Integration configurations and endpoints
- Performance measurement criteria
- Monitoring scope and coverage

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