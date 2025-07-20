"""
Factory for selecting appropriate prompts based on source description
"""

from typing import Dict, Any, Optional
from .extract_nodes import versions as generic_node_versions
from .extract_nodes_github import versions as github_node_versions
from .extract_nodes_cloud import versions as cloud_node_versions
from .extract_nodes_cicd import versions as cicd_node_versions
from .extract_nodes_monitoring import versions as monitoring_node_versions

from .extract_edges import versions as generic_edge_versions
from .extract_edges_github import versions as github_edge_versions
from .extract_edges_cloud import versions as cloud_edge_versions
from .extract_edges_cicd import versions as cicd_edge_versions
from .extract_edges_monitoring import versions as monitoring_edge_versions


class PromptSelector:
    """
    Selects appropriate prompt versions based on exact source description match.
    """
    
    def __init__(self):
        # Registry for node extraction prompts
        self._node_registry: Dict[str, Any] = {
            'github_repo': github_node_versions,
            'cloud_resources': cloud_node_versions,
            'cicd_resources': cicd_node_versions,
            'monitoring_resources': monitoring_node_versions,
        }
        
        # Registry for edge extraction prompts
        self._edge_registry: Dict[str, Any] = {
            'github_repo': github_edge_versions,
            'cloud_resources': cloud_edge_versions,
            'cicd_resources': cicd_edge_versions,
            'monitoring_resources': monitoring_edge_versions,
        }
        
        # Default fallbacks
        self._default_nodes = generic_node_versions
        self._default_edges = generic_edge_versions
    
    def register_node_prompts(self, source_description: str, prompt_versions: Any):
        """Register new source-specific node extraction prompts."""
        self._node_registry[source_description] = prompt_versions
    
    def register_edge_prompts(self, source_description: str, prompt_versions: Any):
        """Register new source-specific edge extraction prompts."""
        self._edge_registry[source_description] = prompt_versions
    
    def get_node_prompts(self, source_description: Optional[str]) -> Any:
        """Get appropriate node extraction prompts based on exact source description match."""
        if not source_description:
            return self._default_nodes
        return self._node_registry.get(source_description, self._default_nodes)
    
    def get_edge_prompts(self, source_description: Optional[str]) -> Any:
        """Get appropriate edge extraction prompts based on exact source description match."""
        if not source_description:
            return self._default_edges
        return self._edge_registry.get(source_description, self._default_edges)


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
