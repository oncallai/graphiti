"""
AWS-specific prompts for edge extraction
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
            content='You are an expert at extracting AWS infrastructure relationships. '
            'Focus on AWS-specific services, networking, and resource dependencies.',
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
Extract relationships between AWS infrastructure entities.

# AWS-SPECIFIC RELATIONSHIPS TO EXTRACT:

1. **EC2 & Compute Relationships**:
   - RESIDES_IN: Instance location (ec2_instance RESIDES_IN vpc)
   - ATTACHED_TO: ENI attachment (eni ATTACHED_TO ec2_instance)
   - RUNS_ON: Container execution (ecs_task RUNS_ON ecs_cluster)
   - SCALED_BY: Auto-scaling (ec2_instance SCALED_BY auto_scaling_group)
   - LOAD_BALANCED_BY: Load balancing (ec2_instance LOAD_BALANCED_BY alb)
   - MANAGED_BY: Systems Manager (ec2_instance MANAGED_BY ssm)
   - PROVISIONED_BY: Infrastructure as Code (resource PROVISIONED_BY cloudformation)

2. **VPC & Networking**:
   - IN_SUBNET: Subnet membership (ec2_instance IN_SUBNET private_subnet)
   - USES_SECURITY_GROUP: Security association (ec2_instance USES_SECURITY_GROUP sg_web)
   - ROUTES_TO: Route table routing (subnet ROUTES_TO internet_gateway)
   - CONNECTED_VIA: VPC peering (vpc_a CONNECTED_VIA vpc_peering)
   - ATTACHED_TO: Gateway attachment (vpc ATTACHED_TO internet_gateway)
   - NATS_THROUGH: NAT gateway (private_subnet NATS_THROUGH nat_gateway)

3. **Storage & Database**:
   - MOUNTED_ON: EBS volume mounting (ebs_volume MOUNTED_ON ec2_instance)
   - STORES_IN: S3 storage (application STORES_IN s3_bucket)
   - BACKED_BY: RDS backing (application BACKED_BY rds_instance)
   - REPLICATES_TO: RDS replication (primary_rds REPLICATES_TO read_replica)
   - SNAPSHOTS_TO: Backup (ebs_volume SNAPSHOTS_TO s3_bucket)
   - ARCHIVED_TO: Glacier archiving (s3_bucket ARCHIVED_TO glacier_vault)

4. **Load Balancing & Auto Scaling**:
   - TARGETS: ALB targets (alb TARGETS target_group)
   - CONTAINS: Target group membership (target_group CONTAINS ec2_instance)
   - SCALES: Auto scaling (auto_scaling_group SCALES ec2_instance)
   - HEALTH_CHECKED_BY: Health monitoring (target_group HEALTH_CHECKED_BY alb)

5. **Database & Cache**:
   - CLUSTER_MEMBER_OF: RDS cluster (rds_instance CLUSTER_MEMBER_OF aurora_cluster)
   - CACHES_FOR: ElastiCache (elasticache_cluster CACHES_FOR rds_instance)
   - READS_FROM: Read replica (application READS_FROM read_replica)
   - WRITES_TO: Primary database (application WRITES_TO primary_rds)
   - REPLICATES_TO: Cross-region (rds_instance REPLICATES_TO dr_region)

6. **Security & IAM**:
   - ASSUMES_ROLE: IAM role assumption (ec2_instance ASSUMES_ROLE iam_role)
   - GRANTS_ACCESS_TO: Permission grants (iam_role GRANTS_ACCESS_TO s3_bucket)
   - ENCRYPTED_BY: KMS encryption (ebs_volume ENCRYPTED_BY kms_key)
   - AUTHENTICATED_BY: Cognito auth (api_gateway AUTHENTICATED_BY cognito_user_pool)
   - AUTHORIZED_BY: IAM policies (resource AUTHORIZED_BY iam_policy)

7. **Multi-Region & Availability**:
   - REPLICATED_IN: Cross-region replication (s3_bucket REPLICATED_IN us-west-2)
   - FAILED_OVER_TO: DR relationships (primary_region FAILED_OVER_TO secondary_region)
   - DISTRIBUTED_ACROSS: Multi-AZ (rds_cluster DISTRIBUTED_ACROSS availability_zones)
   - BACKED_UP_TO: Cross-region backup (rds_instance BACKED_UP_TO backup_region)

8. **Lambda & Serverless**:
   - TRIGGERED_BY: Event triggers (lambda_function TRIGGERED_BY s3_event)
   - INVOKES: Function invocation (api_gateway INVOKES lambda_function)
   - PUBLISHES_TO: SNS publishing (lambda_function PUBLISHES_TO sns_topic)
   - CONSUMES_FROM: SQS consumption (lambda_function CONSUMES_FROM sqs_queue)

# EXTRACTION RULES FOR AWS:

1. Focus on AWS-specific service names and relationships
2. Include region/AZ information when relevant
3. Capture both active and standby relationships
4. Extract from:
   - CloudFormation templates
   - Terraform AWS provider
   - AWS CLI output
   - AWS Console configurations
   - Architecture diagrams

{context['custom_prompt']}

# EXAMPLES:
- "EC2 instances in prod-vpc" → ec2_instance RESIDES_IN prod-vpc
- "ALB distributes traffic to target group" → alb TARGETS target_group
- "RDS with read replicas" → read_replica READS_FROM primary_rds
- "S3 bucket encrypted with KMS" → s3_bucket ENCRYPTED_BY kms_key
- "Lambda triggered by S3 upload" → lambda_function TRIGGERED_BY s3_event
        """,
        ),
    ]


def reflexion(context: dict[str, Any]) -> list[Message]:
    sys_prompt = """You are an AI assistant that determines which AWS infrastructure relationships have not been extracted from the given context"""

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

Given the above AWS infrastructure context, check for missing:
- EC2 & Compute relationships (RESIDES_IN, ATTACHED_TO, RUNS_ON, SCALED_BY)
- VPC & Networking relationships (IN_SUBNET, USES_SECURITY_GROUP, ROUTES_TO)
- Storage & Database relationships (MOUNTED_ON, STORES_IN, BACKED_BY)
- Load Balancing relationships (TARGETS, CONTAINS, SCALES)
- Security & IAM relationships (ASSUMES_ROLE, GRANTS_ACCESS_TO, ENCRYPTED_BY)
- Lambda & Serverless relationships (TRIGGERED_BY, INVOKES, PUBLISHES_TO)

Focus on AWS-specific services and infrastructure topology.
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt),
    ]


def extract_attributes(context: dict[str, Any]) -> list[Message]:
    return [
        Message(
            role='system',
            content='You are a helpful assistant that extracts AWS infrastructure relationship properties.',
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

Given the above AWS infrastructure content and the following relationship, update attributes based on:
- AWS region/AZ information
- VPC CIDR blocks or subnet IPs
- EC2 instance types or sizes
- KMS encryption details
- RDS replication configurations
- Auto-scaling group settings
- Security group rules
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