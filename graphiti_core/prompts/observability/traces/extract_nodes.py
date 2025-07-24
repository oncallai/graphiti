"""
Observability Traces-specific prompts for node extraction
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
    sys_prompt = """You are an AI assistant specialized in extracting distributed tracing entities.
    Your primary focus is identifying tracing systems, trace collection tools, and trace analysis platforms."""

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

You are analyzing distributed tracing systems. Extract entities focusing on:

## PRIMARY EXTRACTION TARGETS:

1. **Distributed Tracing Platforms**:
   - Jaeger distributed tracing system
   - Zipkin distributed tracing
   - AWS X-Ray tracing service
   - Azure Application Insights
   - Google Cloud Trace
   - Datadog APM tracing
   - New Relic distributed tracing
   - OpenTelemetry tracing
   - Lightstep tracing platform
   - Honeycomb tracing

2. **Trace Collection & Instrumentation**:
   - Trace collectors and agents
   - Application instrumentation libraries
   - Auto-instrumentation tools
   - Manual instrumentation code
   - Trace sampling configurations
   - Trace context propagation
   - Trace correlation systems
   - Trace enrichment tools

3. **Trace Storage & Processing**:
   - Trace storage systems and databases
   - Trace indexing and search engines
   - Trace compression and optimization
   - Trace retention and archival systems
   - Trace backup and recovery
   - Trace security and access control
   - Trace federation and sharing
   - Trace data lakes and warehouses

4. **Trace Analysis & Visualization**:
   - Trace analysis dashboards
   - Trace visualization tools
   - Trace query and search interfaces
   - Trace comparison and diff tools
   - Trace performance analysis
   - Trace dependency mapping
   - Trace anomaly detection
   - Trace machine learning tools

5. **Service Mesh & Network Tracing**:
   - Service mesh platforms (Istio, Linkerd, Consul)
   - Network tracing tools
   - Load balancer tracing
   - API gateway tracing
   - Proxy tracing (Envoy, HAProxy)
   - Network flow analysis
   - Traffic routing tracing
   - Circuit breaker tracing

6. **Database & Storage Tracing**:
   - Database query tracing
   - Storage operation tracing
   - Cache tracing and analysis
   - Message queue tracing
   - Event streaming tracing
   - File system operation tracing
   - Backup and restore tracing
   - Data migration tracing

7. **Security & Compliance Tracing**:
   - Security audit tracing
   - Compliance monitoring tracing
   - Access control tracing
   - Authentication tracing
   - Authorization tracing
   - Data access tracing
   - Privacy compliance tracing
   - Forensic analysis tracing

8. **Custom Trace Applications**:
   - Business process tracing
   - User journey tracing
   - Workflow tracing
   - Custom trace collectors
   - Trace correlation with business metrics
   - Custom trace analysis tools
   - Trace-based alerting
   - Trace-driven automation

## TRACING-SPECIFIC RULES:
- Extract tracing platform and system names
- Include trace collection agent and tool names
- Capture trace analysis and visualization tool names
- Extract from tracing configurations and manifests
- Include environment names (prod, staging, dev)
- Capture service mesh and network tracing names

## TRACING NAMING CONVENTIONS:
- Tracing platforms: use platform names (e.g., "jaeger-cluster", "zipkin-instance")
- Trace collectors: use collector names (e.g., "jaeger-agent", "otlp-collector")
- Trace dashboards: use dashboard names (e.g., "service-trace-dashboard", "latency-analysis")
- Trace services: use service names (e.g., "user-service-traces", "payment-api-traces")

## DO NOT EXTRACT:
- Generic trace names without context
- Temporary trace files or artifacts
- Trace configuration values or parameters
- Personal information or credentials
- Generic tracing terms without specific instances

{context['custom_prompt']}
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt),
    ]


def extract_json(context: dict[str, Any]) -> list[Message]:
    sys_prompt = """You are an AI assistant specialized in extracting tracing entities from JSON configurations.
    Focus on tracing platform configurations, trace collection settings, and trace analysis tool configurations."""

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

You are analyzing JSON from tracing systems. This could be:
- Tracing platform configurations
- Trace collection agent settings
- Trace analysis tool configurations
- Service mesh configurations
- Trace storage settings
- Trace visualization configurations

## EXTRACTION FOCUS FOR TRACING JSON:

1. **From Tracing Platform Configs**:
   - Platform names and cluster names
   - Trace collector and agent names
   - Storage and backend system names
   - Query and search interface names
   - Integration and API endpoint names

2. **From Trace Collection Configs**:
   - Collection agent and tool names
   - Instrumentation library names
   - Sampling and filtering configuration names
   - Context propagation system names
   - Trace correlation tool names

3. **From Service Mesh Configs**:
   - Service mesh platform names
   - Proxy and sidecar names
   - Traffic management rule names
   - Security and policy names
   - Observability integration names

4. **From Trace Analysis Configs**:
   - Analysis dashboard and tool names
   - Query and search interface names
   - Visualization and chart names
   - Alert and notification names
   - Export and integration names

## TRACING JSON SPECIFIC RULES:
- Extract tracing platform names from appropriate fields
- Include trace collection agent and tool names
- Capture trace analysis and visualization tool names
- Focus on service mesh and network tracing names
- Skip generic configuration values

{context['custom_prompt']}
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt),
    ]


def extract_text(context: dict[str, Any]) -> list[Message]:
    sys_prompt = """You are an AI assistant specialized in extracting tracing entities from documentation and reports.
    Focus on tracing documentation, trace analysis reports, and distributed tracing system descriptions."""

    user_prompt = f"""
<TEXT>
{context['episode_content']}
</TEXT>
<ENTITY TYPES>
{context['entity_types']}
</ENTITY TYPES>

You are analyzing text about tracing systems. This could be:
- Tracing system documentation
- Trace analysis reports
- Tracing configuration guides
- Trace visualization dashboards
- Tracing troubleshooting guides
- Distributed tracing architecture documentation

## EXTRACTION FOCUS FOR TRACING TEXT:

1. **Tracing Platform Information**:
   - Platform names and descriptions
   - Cluster and system names
   - Collector and agent names
   - Storage and backend system names
   - Query and search interface names

2. **Trace Collection Information**:
   - Collection agent and tool names
   - Instrumentation library names
   - Sampling and filtering system names
   - Context propagation system names
   - Trace correlation tool names

3. **Service Mesh Information**:
   - Service mesh platform names
   - Proxy and sidecar names
   - Traffic management rule names
   - Security and policy names
   - Observability integration names

4. **Trace Analysis Information**:
   - Analysis dashboard and tool names
   - Query and search interface names
   - Visualization and chart names
   - Alert and notification names
   - Export and integration names

## TRACING TEXT SPECIFIC RULES:
- Extract specific tracing platform and tool names
- Include environment prefixes/suffixes (prod-, -staging)
- Capture trace collection agent and system names
- Focus on currently used tracing tools and platforms
- Extract from configuration examples and reports

## TRACING NAMING STANDARDS:
- Use full tracing platform names (e.g., "jaeger-production-cluster")
- Include environment context (e.g., "staging-service-traces")
- Maintain original naming conventions
- Use canonical names for well-known tracing platforms

{context['custom_prompt']}
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt),
    ]


def reflexion(context: dict[str, Any]) -> list[Message]:
    sys_prompt = """You are an AI assistant that determines which tracing entities have not been extracted from the given context"""

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

Given the above previous messages, current message, and list of extracted entities; determine if any tracing entities haven't been extracted.

Focus on missing:
- Distributed tracing platforms and systems
- Trace collection and instrumentation tools
- Trace storage and processing systems
- Trace analysis and visualization tools
- Service mesh and network tracing tools
- Database and storage tracing systems
- Security and compliance tracing tools
- Custom trace applications and tools
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt),
    ]


def classify_nodes(context: dict[str, Any]) -> list[Message]:
    sys_prompt = """You are an AI assistant that classifies tracing entity nodes given the context from which they were extracted"""

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
    
    Given the above conversation, extracted entities, and provided entity types and their descriptions, classify the extracted tracing entities.
    
    Guidelines:
    1. Each entity must have exactly one type
    2. Only use the provided ENTITY TYPES as types, do not use additional types to classify entities.
    3. If none of the provided entity types accurately classify an extracted node, the type should be set to None
    4. Consider tracing-specific context when classifying (e.g., tracing platforms, trace collectors, service mesh tools)
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt),
    ]


def extract_attributes(context: dict[str, Any]) -> list[Message]:
    return [
        Message(
            role='system',
            content='You are a helpful assistant that extracts tracing entity properties from the provided text.',
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
        4. Consider tracing-specific attributes like sampling rates, trace formats, retention policies, etc.
        
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