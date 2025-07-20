"""
Cloud resources specific prompts for edge extraction
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
            content='You are an expert at extracting infrastructure relationships from cloud resources. '
            'Focus on network topology, security relationships, and resource dependencies.',
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
Extract relationships between cloud infrastructure entities.

# CLOUD-SPECIFIC RELATIONSHIPS TO EXTRACT:

1. **Network Relationships**:
   - RESIDES_IN: Resource location (instance RESIDES_IN vpc)
   - ATTACHED_TO: Network attachment (eni ATTACHED_TO instance)
   - ROUTES_TO: Traffic routing (alb ROUTES_TO target_group)
   - CONNECTED_VIA: Network connectivity (vpc CONNECTED_VIA peering)
   - IN_SUBNET: Subnet membership (instance IN_SUBNET subnet)
   - USES_SECURITY_GROUP: Security association (instance USES_SECURITY_GROUP sg)

2. **Storage Relationships**:
   - MOUNTED_ON: Volume mounting (ebs_volume MOUNTED_ON instance)
   - STORES_IN: Storage location (application STORES_IN s3_bucket)
   - BACKED_BY: Storage backing (database BACKED_BY ebs_volume)
   - REPLICATES_TO: Replication (primary_db REPLICATES_TO replica_db)
   - SNAPSHOTS_TO: Backup relationships (volume SNAPSHOTS_TO s3)

3. **Compute Relationships**:
   - RUNS_ON: Execution location (container RUNS_ON ecs_cluster)
   - SCALED_BY: Auto-scaling (instance_group SCALED_BY auto_scaler)
   - LOAD_BALANCED_BY: Load balancing (instance LOAD_BALANCED_BY alb)
   - MANAGED_BY: Management relationships (instance MANAGED_BY ssm)
   - PROVISIONED_BY: Provisioning (resource PROVISIONED_BY cloudformation)

4. **Database & Cache**:
   - CLUSTER_MEMBER_OF: Cluster membership (node CLUSTER_MEMBER_OF rds_cluster)
   - CACHES_FOR: Cache relationships (elasticache CACHES_FOR rds)
   - READS_FROM: Read replica (app READS_FROM read_replica)
   - WRITES_TO: Write relationships (app WRITES_TO primary_db)

5. **Security & IAM**:
   - ASSUMES_ROLE: IAM relationships (instance ASSUMES_ROLE iam_role)
   - GRANTS_ACCESS_TO: Permission grants (role GRANTS_ACCESS_TO s3_bucket)
   - ENCRYPTED_BY: Encryption (volume ENCRYPTED_BY kms_key)
   - AUTHENTICATED_BY: Auth relationships (api AUTHENTICATED_BY cognito)

6. **Multi-Region/AZ**:
   - REPLICATED_IN: Cross-region replication (bucket REPLICATED_IN region)
   - FAILED_OVER_TO: DR relationships (primary FAILED_OVER_TO secondary)
   - DISTRIBUTED_ACROSS: Distribution (cluster DISTRIBUTED_ACROSS availability_zones)

# EXTRACTION RULES FOR CLOUD:

1. Focus on infrastructure topology and dependencies
2. Include region/AZ information when relevant
3. Capture both active and standby relationships
4. Extract from:
   - Terraform/CloudFormation templates
   - Cloud provider documentation
   - Architecture diagrams
   - Network configurations
   - Security policies

{context['custom_prompt']}

# EXAMPLES:
- "EC2 instances in prod-vpc" → instance RESIDES_IN prod-vpc
- "ALB distributes traffic to ASG" → asg LOAD_BALANCED_BY alb
- "RDS with read replicas" → read_replica READS_FROM primary_db
- "S3 bucket encrypted with KMS" → s3_bucket ENCRYPTED_BY kms_key
        """,
        ),
    ]


def reflexion(context: dict[str, Any]) -> list[Message]:
    sys_prompt = """You are an AI assistant that determines which cloud infrastructure relationships have not been extracted from the given context"""

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

Given the above cloud infrastructure context, check for missing:
- Network relationships (RESIDES_IN, ATTACHED_TO, ROUTES_TO)
- Storage relationships (MOUNTED_ON, STORES_IN, BACKED_BY)
- Compute relationships (RUNS_ON, SCALED_BY, LOAD_BALANCED_BY)
- Security relationships (ASSUMES_ROLE, ENCRYPTED_BY)
- Multi-region/AZ relationships (REPLICATED_IN, DISTRIBUTED_ACROSS)

Focus on infrastructure topology and dependencies.
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt),
    ]


def extract_attributes(context: dict[str, Any]) -> list[Message]:
    return [
        Message(
            role='system',
            content='You are a helpful assistant that extracts cloud infrastructure relationship properties.',
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

Given the above cloud infrastructure content and the following relationship, update attributes based on:
- Region/AZ information
- Network CIDR blocks or IPs
- Instance types or sizes
- Encryption details
- Replication configurations
- High availability settings

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