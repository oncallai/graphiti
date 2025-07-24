"""
Azure-specific prompts for edge extraction
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
            content='You are an expert at extracting Azure infrastructure relationships. '
            'Focus on Azure-specific services, networking, and resource dependencies.',
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
Extract relationships between Azure infrastructure entities.

# AZURE-SPECIFIC RELATIONSHIPS TO EXTRACT:

1. **Virtual Machines & Compute**:
   - RESIDES_IN: VM location (vm RESIDES_IN virtual_network)
   - ATTACHED_TO: NIC attachment (network_interface ATTACHED_TO vm)
   - RUNS_ON: Container execution (container_instance RUNS_ON container_group)
   - SCALED_BY: VMSS scaling (vm SCALED_BY virtual_machine_scale_set)
   - LOAD_BALANCED_BY: Load balancing (vm LOAD_BALANCED_BY load_balancer)
   - MANAGED_BY: Azure Arc (vm MANAGED_BY arc_agent)
   - PROVISIONED_BY: ARM templates (resource PROVISIONED_BY arm_template)

2. **Virtual Networks & Networking**:
   - IN_SUBNET: Subnet membership (vm IN_SUBNET private_subnet)
   - USES_NSG: Network security group (vm USES_NSG network_security_group)
   - ROUTES_TO: Route table routing (subnet ROUTES_TO internet_gateway)
   - CONNECTED_VIA: VNet peering (vnet_a CONNECTED_VIA vnet_peering)
   - ATTACHED_TO: Gateway attachment (vnet ATTACHED_TO virtual_network_gateway)
   - NATS_THROUGH: NAT gateway (private_subnet NATS_THROUGH nat_gateway)

3. **Storage & Database**:
   - MOUNTED_ON: Managed disk mounting (managed_disk MOUNTED_ON vm)
   - STORES_IN: Blob storage (application STORES_IN storage_account)
   - BACKED_BY: SQL Database backing (application BACKED_BY sql_database)
   - REPLICATES_TO: SQL replication (primary_sql REPLICATES_TO read_replica)
   - SNAPSHOTS_TO: Backup (managed_disk SNAPSHOTS_TO recovery_services_vault)
   - ARCHIVED_TO: Archive tier (storage_account ARCHIVED_TO archive_tier)

4. **Load Balancing & Availability**:
   - TARGETS: Load balancer targets (load_balancer TARGETS backend_pool)
   - CONTAINS: Backend pool membership (backend_pool CONTAINS vm)
   - SCALES: VMSS scaling (virtual_machine_scale_set SCALES vm)
   - HEALTH_CHECKED_BY: Health monitoring (backend_pool HEALTH_CHECKED_BY load_balancer)
   - DISTRIBUTED_ACROSS: Availability zones (vmss DISTRIBUTED_ACROSS availability_zones)

5. **Database & Cache**:
   - CLUSTER_MEMBER_OF: SQL cluster (sql_database CLUSTER_MEMBER_OF elastic_pool)
   - CACHES_FOR: Redis Cache (redis_cache CACHES_FOR sql_database)
   - READS_FROM: Read replica (application READS_FROM read_replica)
   - WRITES_TO: Primary database (application WRITES_TO primary_sql)
   - REPLICATES_TO: Geo-replication (sql_database REPLICATES_TO secondary_region)

6. **Security & Identity**:
   - ASSUMES_ROLE: Managed identity (vm ASSUMES_ROLE managed_identity)
   - GRANTS_ACCESS_TO: RBAC grants (role_definition GRANTS_ACCESS_TO storage_account)
   - ENCRYPTED_BY: Key Vault encryption (managed_disk ENCRYPTED_BY key_vault)
   - AUTHENTICATED_BY: Azure AD auth (api_management AUTHENTICATED_BY azure_ad)
   - AUTHORIZED_BY: RBAC policies (resource AUTHORIZED_BY role_assignment)

7. **Multi-Region & Disaster Recovery**:
   - REPLICATED_IN: Geo-replication (storage_account REPLICATED_IN west-us-2)
   - FAILED_OVER_TO: DR relationships (primary_region FAILED_OVER_TO secondary_region)
   - BACKED_UP_TO: Cross-region backup (sql_database BACKED_UP_TO backup_region)
   - DISTRIBUTED_ACROSS: Multi-region (cosmos_db DISTRIBUTED_ACROSS regions)

8. **Functions & Serverless**:
   - TRIGGERED_BY: Event triggers (function_app TRIGGERED_BY storage_blob)
   - INVOKES: Function invocation (api_management INVOKES function_app)
   - PUBLISHES_TO: Event Grid (function_app PUBLISHES_TO event_grid_topic)
   - CONSUMES_FROM: Service Bus (function_app CONSUMES_FROM service_bus_queue)

9. **App Service & Web Apps**:
   - HOSTED_ON: App Service hosting (web_app HOSTED_ON app_service_plan)
   - CONNECTS_TO: Database connections (web_app CONNECTS_TO sql_database)
   - DEPLOYS_TO: Deployment slots (web_app DEPLOYS_TO staging_slot)
   - MONITORED_BY: Application Insights (web_app MONITORED_BY application_insights)

# EXTRACTION RULES FOR AZURE:

1. Focus on Azure-specific service names and relationships
2. Include region/AZ information when relevant
3. Capture both active and standby relationships
4. Extract from:
   - ARM templates
   - Terraform Azure provider
   - Azure CLI output
   - Azure Portal configurations
   - Architecture diagrams

{context['custom_prompt']}

# EXAMPLES:
- "VM in prod-vnet" → vm RESIDES_IN prod-vnet
- "Load balancer targets backend pool" → load_balancer TARGETS backend_pool
- "SQL with read replicas" → read_replica READS_FROM primary_sql
- "Storage account encrypted with Key Vault" → storage_account ENCRYPTED_BY key_vault
- "Function triggered by blob storage" → function_app TRIGGERED_BY storage_blob
        """,
        ),
    ]


def reflexion(context: dict[str, Any]) -> list[Message]:
    sys_prompt = """You are an AI assistant that determines which Azure infrastructure relationships have not been extracted from the given context"""

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

Given the above Azure infrastructure context, check for missing:
- VM & Compute relationships (RESIDES_IN, ATTACHED_TO, RUNS_ON, SCALED_BY)
- VNet & Networking relationships (IN_SUBNET, USES_NSG, ROUTES_TO)
- Storage & Database relationships (MOUNTED_ON, STORES_IN, BACKED_BY)
- Load Balancing relationships (TARGETS, CONTAINS, SCALES)
- Security & Identity relationships (ASSUMES_ROLE, GRANTS_ACCESS_TO, ENCRYPTED_BY)
- Functions & Serverless relationships (TRIGGERED_BY, INVOKES, PUBLISHES_TO)
- App Service relationships (HOSTED_ON, CONNECTS_TO, MONITORED_BY)

Focus on Azure-specific services and infrastructure topology.
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt),
    ]


def extract_attributes(context: dict[str, Any]) -> list[Message]:
    return [
        Message(
            role='system',
            content='You are a helpful assistant that extracts Azure infrastructure relationship properties.',
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

Given the above Azure infrastructure content and the following relationship, update attributes based on:
- Azure region/AZ information
- VNet CIDR blocks or subnet IPs
- VM sizes or SKUs
- Key Vault encryption details
- SQL Database replication configurations
- VMSS scaling settings
- NSG security rules
- RBAC role permissions

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