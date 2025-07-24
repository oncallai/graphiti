"""
Observability Traces-specific prompts for edge extraction
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
            content='You are an expert at extracting distributed tracing relationships. '
            'Focus on trace flow dependencies, service call relationships, and distributed system connections.',
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
Extract relationships between distributed tracing entities.

# TRACING-SPECIFIC RELATIONSHIPS TO EXTRACT:

1. **Trace Collection & Flow Relationships**:
   - COLLECTS_FROM: Trace collection (jaeger COLLECTS_FROM application)
   - EXPORTS_TO: Trace export (application EXPORTS_TO jaeger)
   - INSTRUMENTS: Application instrumentation (agent INSTRUMENTS service)
   - SAMPLES: Trace sampling (collector SAMPLES traces)
   - PROPAGATES: Context propagation (service PROPAGATES trace_context)
   - CORRELATES: Trace correlation (tool CORRELATES spans)

2. **Service Call & Dependency Relationships**:
   - CALLS: Service calls (api_service CALLS database_service)
   - DEPENDS_ON: Service dependencies (frontend DEPENDS_ON backend)
   - COMMUNICATES_WITH: Service communication (service_a COMMUNICATES_WITH service_b)
   - INVOKES: Function invocation (lambda INVOKES external_api)
   - QUERIES: Database queries (service QUERIES database)
   - PUBLISHES_TO: Message publishing (service PUBLISHES_TO queue)

3. **Service Mesh & Network Relationships**:
   - ROUTES_THROUGH: Traffic routing (request ROUTES_THROUGH istio_proxy)
   - LOAD_BALANCES: Load balancing (load_balancer LOAD_BALANCES services)
   - PROXIES: Request proxying (envoy PROXIES api_gateway)
   - CIRCUIT_BREAKS: Circuit breaking (circuit_breaker CIRCUIT_BREAKS service)
   - RATE_LIMITS: Rate limiting (rate_limiter RATE_LIMITS requests)
   - AUTHENTICATES: Authentication (auth_service AUTHENTICATES requests)

4. **Trace Analysis & Visualization Relationships**:
   - ANALYZES: Trace analysis (jaeger ANALYZES service_traces)
   - VISUALIZES: Trace visualization (dashboard VISUALIZES trace_flow)
   - QUERIES: Trace querying (ui QUERIES trace_database)
   - COMPARES: Trace comparison (tool COMPARES trace_performance)
   - DETECTS: Anomaly detection (system DETECTS slow_traces)
   - ALERTS_ON: Trace alerting (alert_rule ALERTS_ON trace_anomaly)

5. **Database & Storage Trace Relationships**:
   - EXECUTES_QUERY: Query execution (service EXECUTES_QUERY sql)
   - READS_FROM: Data reading (service READS_FROM cache)
   - WRITES_TO: Data writing (service WRITES_TO database)
   - BACKS_UP: Data backup (backup_service BACKS_UP trace_data)
   - ARCHIVES: Data archival (archiver ARCHIVES old_traces)
   - REPLICATES: Data replication (replicator REPLICATES trace_data)

6. **Security & Compliance Trace Relationships**:
   - AUDITS: Security auditing (auditor AUDITS access_traces)
   - MONITORS: Security monitoring (security_monitor MONITORS trace_access)
   - VALIDATES: Access validation (validator VALIDATES user_permissions)
   - ENCRYPTS: Data encryption (encryptor ENCRYPTS sensitive_traces)
   - COMPLIES_WITH: Compliance checking (system COMPLIES_WITH privacy_policy)
   - DETECTS_THREAT: Threat detection (threat_detector DETECTS_THREAT suspicious_trace)

7. **Trace Storage & Processing Relationships**:
   - STORES_IN: Trace storage (jaeger STORES_IN elasticsearch)
   - INDEXES: Trace indexing (indexer INDEXES trace_data)
   - COMPRESSES: Trace compression (compressor COMPRESSES trace_files)
   - DEDUPLICATES: Trace deduplication (deduplicator DEDUPLICATES traces)
   - ENRICHES: Trace enrichment (enricher ENRICHES with_metadata)
   - TRANSFORMS: Trace transformation (transformer TRANSFORMS trace_format)

8. **Integration & API Trace Relationships**:
   - INTEGRATES_WITH: Tool integration (jaeger INTEGRATES_WITH grafana)
   - WEBHOOKS_TO: Webhook relationships (trace_event WEBHOOKS_TO notification)
   - CALLS_API: API calling (collector CALLS_API trace_endpoint)
   - SUBSCRIBES_TO: Event subscription (analyzer SUBSCRIBES_TO trace_events)
   - PUBLISHES_EVENT: Event publishing (service PUBLISHES_EVENT trace_complete)
   - SYNCHRONIZES_WITH: Data synchronization (jaeger SYNCHRONIZES_WITH zipkin)

9. **Business Process Trace Relationships**:
   - TRACKS_JOURNEY: User journey tracking (tracer TRACKS_JOURNEY user_flow)
   - MONITORS_WORKFLOW: Workflow monitoring (monitor MONITORS_WORKFLOW process)
   - CORRELATES_BUSINESS: Business correlation (correlator CORRELATES_BUSINESS metrics)
   - OPTIMIZES_PROCESS: Process optimization (optimizer OPTIMIZES_PROCESS flow)
   - MEASURES_SLA: SLA measurement (sla_monitor MEASURES_SLA performance)
   - TRACKS_KPI: KPI tracking (kpi_tracker TRACKS_KPI business_metrics)

# EXTRACTION RULES FOR TRACING:

1. Focus on distributed trace flow and service call dependencies
2. Include service mesh and network routing relationships
3. Capture trace analysis and visualization data flow relationships
4. Extract from:
   - Tracing platform configurations
   - Service mesh configurations
   - Application instrumentation settings
   - Trace analysis and visualization definitions
   - Distributed system architecture documents

{context['custom_prompt']}

# EXAMPLES:
- "Jaeger collects traces from application" → jaeger COLLECTS_FROM application
- "API service calls database service" → api_service CALLS database_service
- "Istio routes traffic through proxy" → request ROUTES_THROUGH istio_proxy
- "Jaeger analyzes service traces" → jaeger ANALYZES service_traces
- "Service queries database" → service QUERIES database
        """,
        ),
    ]


def reflexion(context: dict[str, Any]) -> list[Message]:
    sys_prompt = """You are an AI assistant that determines which tracing relationships have not been extracted from the given context"""

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

Given the above tracing context, check for missing:
- Trace collection and flow relationships (COLLECTS_FROM, EXPORTS_TO, INSTRUMENTS, SAMPLES)
- Service call and dependency relationships (CALLS, DEPENDS_ON, COMMUNICATES_WITH, INVOKES)
- Service mesh and network relationships (ROUTES_THROUGH, LOAD_BALANCES, PROXIES, CIRCUIT_BREAKS)
- Trace analysis and visualization relationships (ANALYZES, VISUALIZES, QUERIES, COMPARES)
- Database and storage trace relationships (EXECUTES_QUERY, READS_FROM, WRITES_TO, BACKS_UP)
- Security and compliance trace relationships (AUDITS, MONITORS, VALIDATES, ENCRYPTS)
- Trace storage and processing relationships (STORES_IN, INDEXES, COMPRESSES, DEDUPLICATES)
- Integration and API trace relationships (INTEGRATES_WITH, WEBHOOKS_TO, CALLS_API, SUBSCRIBES_TO)
- Business process trace relationships (TRACKS_JOURNEY, MONITORS_WORKFLOW, CORRELATES_BUSINESS)

Focus on distributed trace flow and service call relationships.
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt),
    ]


def extract_attributes(context: dict[str, Any]) -> list[Message]:
    return [
        Message(
            role='system',
            content='You are a helpful assistant that extracts tracing relationship properties.',
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

Given the above tracing content and the following relationship, update attributes based on:
- Trace sampling rates and configurations
- Service call latencies and timeouts
- Trace context propagation settings
- Service mesh routing rules and policies
- Trace storage and retention configurations
- Security and compliance requirements
- Integration endpoints and protocols

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