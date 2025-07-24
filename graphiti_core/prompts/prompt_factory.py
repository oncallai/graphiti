"""
Factory for selecting appropriate prompts based on source description
"""

import logging
from typing import Dict, Any, Optional
from .extract_nodes import versions as generic_node_versions
from .extract_edges import versions as generic_edge_versions

# Cloud provider imports
from .cloud.aws.extract_nodes import versions as aws_node_versions
from .cloud.aws.extract_edges import versions as aws_edge_versions
from .cloud.azure.extract_nodes import versions as azure_node_versions
from .cloud.azure.extract_edges import versions as azure_edge_versions
from .cloud.gcp.extract_nodes import versions as gcp_node_versions
from .cloud.gcp.extract_edges import versions as gcp_edge_versions

# GitHub imports
from .github.extract_nodes import versions as github_node_versions
from .github.extract_edges import versions as github_edge_versions

# CI/CD imports
from .cicd.extract_nodes import versions as cicd_node_versions
from .cicd.extract_edges import versions as cicd_edge_versions

# Observability imports
from .observability.logs.extract_nodes import versions as logs_node_versions
from .observability.metrics.extract_nodes import versions as metrics_node_versions
from .observability.metrics.extract_edges import versions as metrics_edge_versions
from .observability.traces.extract_nodes import versions as traces_node_versions
from .observability.traces.extract_edges import versions as traces_edge_versions

# Application-centric imports
from .extract_nodes_application import versions as application_node_versions
from .extract_edges_application import versions as application_edge_versions


class PromptSelector:
    """
    Selects appropriate prompt versions based on exact source description match.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Domain mapping for better logging
        self._domain_mapping = {
            'aws_resources': 'AWS Cloud Infrastructure',
            'azure_resources': 'Azure Cloud Infrastructure', 
            'gcp_resources': 'GCP Cloud Infrastructure',
            'cloud_resources': 'Generic Cloud Infrastructure',
            'github_repo': 'GitHub Repository & Development',
            'github_resources': 'GitHub Development Tools',
            'cicd_resources': 'CI/CD Pipeline Systems',
            'pipeline_resources': 'Deployment Pipeline Tools',
            'logs_resources': 'Observability Logging Systems',
            'metrics_resources': 'Observability Metrics & Monitoring',
            'traces_resources': 'Observability Distributed Tracing',
            'observability_resources': 'General Observability Systems',
            'monitoring_resources': 'Monitoring & Alerting Systems',
            'application_resources': 'Cross-Domain Application Hub'
        }
        
        # Registry for node extraction prompts
        self._node_registry: Dict[str, Any] = {
            # Cloud providers
            'aws_resources': aws_node_versions,
            'azure_resources': azure_node_versions,
            'gcp_resources': gcp_node_versions,
            'cloud_resources': aws_node_versions,  # Default to AWS for generic cloud
            
            # GitHub
            'github_repo': github_node_versions,
            'github_resources': github_node_versions,
            
            # CI/CD
            'cicd_resources': cicd_node_versions,
            'pipeline_resources': cicd_node_versions,
            
            # Observability
            'logs_resources': logs_node_versions,
            'metrics_resources': metrics_node_versions,
            'traces_resources': traces_node_versions,
            'observability_resources': metrics_node_versions,  # Default to metrics for generic observability
            'monitoring_resources': metrics_node_versions,  # Legacy support
            
            # Application-centric
            'application_resources': application_node_versions,
        }
        
        # Registry for edge extraction prompts
        self._edge_registry: Dict[str, Any] = {
            # Cloud providers
            'aws_resources': aws_edge_versions,
            'azure_resources': azure_edge_versions,
            'gcp_resources': gcp_edge_versions,
            'cloud_resources': aws_edge_versions,  # Default to AWS for generic cloud
            
            # GitHub
            'github_repo': github_edge_versions,
            'github_resources': github_edge_versions,
            
            # CI/CD
            'cicd_resources': cicd_edge_versions,
            'pipeline_resources': cicd_edge_versions,
            
            # Observability
            'metrics_resources': metrics_edge_versions,
            'traces_resources': traces_edge_versions,
            'observability_resources': metrics_edge_versions,  # Default to metrics for generic observability
            'monitoring_resources': metrics_edge_versions,  # Legacy support
            
            # Application-centric
            'application_resources': application_edge_versions,
        }
        
        # Default fallbacks
        self._default_nodes = generic_node_versions
        self._default_edges = generic_edge_versions
        
        # Log the initialization
        self.logger.info("🚀 PromptSelector initialized with domain-specific prompts:")
        for source_desc, domain_name in self._domain_mapping.items():
            has_nodes = source_desc in self._node_registry
            has_edges = source_desc in self._edge_registry
            self.logger.info(f"   📁 {source_desc} → {domain_name} (Nodes: {'✅' if has_nodes else '❌'}, Edges: {'✅' if has_edges else '❌'})")
    
    def register_node_prompts(self, source_description: str, prompt_versions: Any):
        """Register new source-specific node extraction prompts."""
        self._node_registry[source_description] = prompt_versions
        self.logger.info(f"📝 Registered new node prompts for '{source_description}'")
    
    def register_edge_prompts(self, source_description: str, prompt_versions: Any):
        """Register new source-specific edge extraction prompts."""
        self._edge_registry[source_description] = prompt_versions
        self.logger.info(f"📝 Registered new edge prompts for '{source_description}'")
    
    def get_node_prompts(self, source_description: Optional[str]) -> Any:
        """Get appropriate node extraction prompts based on exact source description match."""
        if not source_description:
            self.logger.info("🔍 [NODES] No source_description provided, using DEFAULT generic prompts")
            return self._default_nodes
        
        domain_name = self._domain_mapping.get(source_description, source_description)
        
        if source_description in self._node_registry:
            self.logger.info(f"🎯 [NODES] Found domain-specific prompts for '{source_description}' ({domain_name})")
            return self._node_registry[source_description]
        else:
            self.logger.warning(f"⚠️ [NODES] No domain-specific prompts found for '{source_description}', falling back to DEFAULT prompts")
            return self._default_nodes
    
    def get_edge_prompts(self, source_description: Optional[str]) -> Any:
        """Get appropriate edge extraction prompts based on exact source description match."""
        if not source_description:
            self.logger.info("🔍 [EDGES] No source_description provided, using DEFAULT generic prompts")
            return self._default_edges
        
        domain_name = self._domain_mapping.get(source_description, source_description)
        
        if source_description in self._edge_registry:
            self.logger.info(f"🎯 [EDGES] Found domain-specific prompts for '{source_description}' ({domain_name})")
            return self._edge_registry[source_description]
        else:
            self.logger.warning(f"⚠️ [EDGES] No domain-specific prompts found for '{source_description}', falling back to DEFAULT prompts")
            return self._default_edges


# Global instance
prompt_selector = PromptSelector()


def get_extract_node_prompts(context: Dict[str, Any]) -> Any:
    """Get the appropriate extract_nodes prompts based on source_description in context."""
    source_description = context.get('source_description', '')
    return prompt_selector.get_node_prompts(source_description)


def get_extract_edge_prompts(context: Dict[str, Any]) -> Any:
    """Get the appropriate extract_edges prompts based on source_description in context."""
    source_description = context.get('source_description', '')
    return prompt_selector.get_edge_prompts(source_description)
