"""
GCP-specific prompts for edge extraction
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
            content='You are an expert at extracting GCP infrastructure relationships. '
            'Focus on GCP-specific services, networking, and resource dependencies.',
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
Extract relationships between GCP infrastructure entities.

# GCP-SPECIFIC RELATIONSHIPS TO EXTRACT:

1. **Compute Engine & Compute**:
   - RESIDES_IN: Instance location (compute_instance RESIDES_IN vpc_network)
   - ATTACHED_TO: NIC attachment (network_interface ATTACHED_TO compute_instance)
   - RUNS_ON: Container execution (gke_pod RUNS_ON gke_cluster)
   - SCALED_BY: Instance group scaling (compute_instance SCALED_BY instance_group)
   - LOAD_BALANCED_BY: Load balancing (compute_instance LOAD_BALANCED_BY load_balancer)
   - MANAGED_BY: Cloud Operations (compute_instance MANAGED_BY ops_agent)
   - PROVISIONED_BY: Deployment Manager (resource PROVISIONED_BY deployment_manager)

2. **VPC & Networking**:
   - IN_SUBNET: Subnet membership (compute_instance IN_SUBNET private_subnet)
   - USES_FIREWALL: Firewall rules (compute_instance USES_FIREWALL firewall_rule)
   - ROUTES_TO: Route table routing (subnet ROUTES_TO internet_gateway)
   - CONNECTED_VIA: VPC peering (vpc_a CONNECTED_VIA vpc_peering)
   - ATTACHED_TO: Gateway attachment (vpc ATTACHED_TO cloud_nat)
   - NATS_THROUGH: Cloud NAT (private_subnet NATS_THROUGH cloud_nat)

3. **Storage & Database**:
   - MOUNTED_ON: Persistent disk mounting (persistent_disk MOUNTED_ON compute_instance)
   - STORES_IN: Cloud Storage (application STORES_IN storage_bucket)
   - BACKED_BY: Cloud SQL backing (application BACKED_BY cloud_sql_instance)
   - REPLICATES_TO: SQL replication (primary_sql REPLICATES_TO read_replica)
   - SNAPSHOTS_TO: Backup (persistent_disk SNAPSHOTS_TO storage_bucket)
   - ARCHIVED_TO: Archive class (storage_bucket ARCHIVED_TO archive_class)

4. **Load Balancing & Availability**:
   - TARGETS: Load balancer targets (load_balancer TARGETS backend_service)
   - CONTAINS: Backend service membership (backend_service CONTAINS compute_instance)
   - SCALES: Instance group scaling (instance_group SCALES compute_instance)
   - HEALTH_CHECKED_BY: Health monitoring (backend_service HEALTH_CHECKED_BY health_check)
   - DISTRIBUTED_ACROSS: Availability zones (instance_group DISTRIBUTED_ACROSS zones)

5. **Database & Cache**:
   - CLUSTER_MEMBER_OF: SQL cluster (cloud_sql_instance CLUSTER_MEMBER_OF sql_cluster)
   - CACHES_FOR: Memorystore (memorystore_instance CACHES_FOR cloud_sql_instance)
   - READS_FROM: Read replica (application READS_FROM read_replica)
   - WRITES_TO: Primary database (application WRITES_TO primary_sql)
   - REPLICATES_TO: Cross-region (cloud_sql_instance REPLICATES_TO secondary_region)

6. **Security & IAM**:
   - ASSUMES_ROLE: Service account (compute_instance ASSUMES_ROLE service_account)
   - GRANTS_ACCESS_TO: IAM grants (service_account GRANTS_ACCESS_TO storage_bucket)
   - ENCRYPTED_BY: Cloud KMS encryption (persistent_disk ENCRYPTED_BY kms_key)
   - AUTHENTICATED_BY: Identity Platform (cloud_run AUTHENTICATED_BY identity_platform)
   - AUTHORIZED_BY: IAM policies (resource AUTHORIZED_BY iam_policy)

7. **Multi-Region & Disaster Recovery**:
   - REPLICATED_IN: Cross-region replication (storage_bucket REPLICATED_IN us-west1)
   - FAILED_OVER_TO: DR relationships (primary_region FAILED_OVER_TO secondary_region)
   - BACKED_UP_TO: Cross-region backup (cloud_sql_instance BACKED_UP_TO backup_region)
   - DISTRIBUTED_ACROSS: Multi-region (spanner_instance DISTRIBUTED_ACROSS regions)

8. **Cloud Functions & Serverless**:
   - TRIGGERED_BY: Event triggers (cloud_function TRIGGERED_BY cloud_storage_event)
   - INVOKES: Function invocation (cloud_run INVOKES cloud_function)
   - PUBLISHES_TO: Pub/Sub (cloud_function PUBLISHES_TO pubsub_topic)
   - CONSUMES_FROM: Pub/Sub (cloud_function CONSUMES_FROM pubsub_subscription)

9. **Kubernetes & GKE**:
   - RUNS_ON: Pod execution (gke_pod RUNS_ON gke_node)
   - DEPLOYED_TO: Deployment (gke_pod DEPLOYED_TO gke_deployment)
   - EXPOSED_BY: Service exposure (gke_pod EXPOSED_BY gke_service)
   - STORED_IN: ConfigMap/Secret (gke_pod STORED_IN config_map)
   - MONITORED_BY: Cloud Monitoring (gke_cluster MONITORED_BY cloud_monitoring)

10. **App Engine & Cloud Run**:
    - HOSTED_ON: App Engine hosting (app_engine_app HOSTED_ON app_engine_service)
    - CONNECTS_TO: Database connections (app_engine_app CONNECTS_TO cloud_sql_instance)
    - DEPLOYS_TO: Version deployment (app_engine_app DEPLOYS_TO app_engine_version)
    - MONITORED_BY: Cloud Monitoring (app_engine_app MONITORED_BY cloud_monitoring)

# EXTRACTION RULES FOR GCP:

1. Focus on GCP-specific service names and relationships
2. Include region/zone information when relevant
3. Capture both active and standby relationships
4. Extract from:
   - Deployment Manager templates
   - Terraform GCP provider
   - gcloud CLI output
   - GCP Console configurations
   - Architecture diagrams

{context['custom_prompt']}

# EXAMPLES:
- "Compute instance in prod-vpc" → compute_instance RESIDES_IN prod-vpc
- "Load balancer targets backend service" → load_balancer TARGETS backend_service
- "Cloud SQL with read replicas" → read_replica READS_FROM primary_sql
- "Storage bucket encrypted with KMS" → storage_bucket ENCRYPTED_BY kms_key
- "Cloud Function triggered by Storage event" → cloud_function TRIGGERED_BY cloud_storage_event
        """,
        ),
    ]


def reflexion(context: dict[str, Any]) -> list[Message]:
    sys_prompt = """You are an AI assistant that determines which GCP infrastructure relationships have not been extracted from the given context"""

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

Given the above GCP infrastructure context, check for missing:
- Compute Engine relationships (RESIDES_IN, ATTACHED_TO, RUNS_ON, SCALED_BY)
- VPC & Networking relationships (IN_SUBNET, USES_FIREWALL, ROUTES_TO)
- Storage & Database relationships (MOUNTED_ON, STORES_IN, BACKED_BY)
- Load Balancing relationships (TARGETS, CONTAINS, SCALES)
- Security & IAM relationships (ASSUMES_ROLE, GRANTS_ACCESS_TO, ENCRYPTED_BY)
- Cloud Functions & Serverless relationships (TRIGGERED_BY, INVOKES, PUBLISHES_TO)
- Kubernetes & GKE relationships (RUNS_ON, DEPLOYED_TO, EXPOSED_BY)
- App Engine & Cloud Run relationships (HOSTED_ON, CONNECTS_TO, MONITORED_BY)

Focus on GCP-specific services and infrastructure topology.
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt),
    ]


def extract_attributes(context: dict[str, Any]) -> list[Message]:
    return [
        Message(
            role='system',
            content='You are a helpful assistant that extracts GCP infrastructure relationship properties.',
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

Given the above GCP infrastructure content and the following relationship, update attributes based on:
- GCP region/zone information
- VPC CIDR blocks or subnet IPs
- Compute instance machine types
- Cloud KMS encryption details
- Cloud SQL replication configurations
- Instance group scaling settings
- Firewall rule configurations
- IAM role permissions

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