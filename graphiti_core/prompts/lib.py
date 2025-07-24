"""
Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import logging
from typing import Any, Protocol, TypedDict

from .dedupe_edges import Prompt as DedupeEdgesPrompt
from .dedupe_edges import Versions as DedupeEdgesVersions
from .dedupe_edges import versions as dedupe_edges_versions
from .dedupe_nodes import Prompt as DedupeNodesPrompt
from .dedupe_nodes import Versions as DedupeNodesVersions
from .dedupe_nodes import versions as dedupe_nodes_versions
from .eval import Prompt as EvalPrompt
from .eval import Versions as EvalVersions
from .eval import versions as eval_versions
from .extract_edge_dates import Prompt as ExtractEdgeDatesPrompt
from .extract_edge_dates import Versions as ExtractEdgeDatesVersions
from .extract_edge_dates import versions as extract_edge_dates_versions
from .extract_edges import Prompt as ExtractEdgesPrompt
from .extract_edges import Versions as ExtractEdgesVersions
from .extract_edges import versions as extract_edges_versions
from .extract_nodes import Prompt as ExtractNodesPrompt
from .extract_nodes import Versions as ExtractNodesVersions
from .extract_nodes import versions as extract_nodes_versions
from .invalidate_edges import Prompt as InvalidateEdgesPrompt
from .invalidate_edges import Versions as InvalidateEdgesVersions
from .invalidate_edges import versions as invalidate_edges_versions
from .models import Message, PromptFunction
from .prompt_helpers import DO_NOT_ESCAPE_UNICODE
from .summarize_nodes import Prompt as SummarizeNodesPrompt
from .summarize_nodes import Versions as SummarizeNodesVersions
from .summarize_nodes import versions as summarize_nodes_versions
from .prompt_factory import get_extract_node_prompts, get_extract_edge_prompts


class PromptLibrary(Protocol):
    extract_nodes: ExtractNodesPrompt
    dedupe_nodes: DedupeNodesPrompt
    extract_edges: ExtractEdgesPrompt
    dedupe_edges: DedupeEdgesPrompt
    invalidate_edges: InvalidateEdgesPrompt
    extract_edge_dates: ExtractEdgeDatesPrompt
    summarize_nodes: SummarizeNodesPrompt
    eval: EvalPrompt


class PromptLibraryImpl(TypedDict):
    extract_nodes: ExtractNodesVersions
    dedupe_nodes: DedupeNodesVersions
    extract_edges: ExtractEdgesVersions
    dedupe_edges: DedupeEdgesVersions
    invalidate_edges: InvalidateEdgesVersions
    extract_edge_dates: ExtractEdgeDatesVersions
    summarize_nodes: SummarizeNodesVersions
    eval: EvalVersions


class VersionWrapper:
    def __init__(self, func: PromptFunction):
        self.func = func

    def __call__(self, context: dict[str, Any]) -> list[Message]:
        messages = self.func(context)
        for message in messages:
            message.content += DO_NOT_ESCAPE_UNICODE if message.role == 'system' else ''
        return messages


class PromptTypeWrapper:
    def __init__(self, versions: dict[str, PromptFunction]):
        for version, func in versions.items():
            setattr(self, version, VersionWrapper(func))


class DynamicPromptTypeWrapper:
    """
    A wrapper that dynamically selects prompts based on context.
    Maintains the same interface as PromptTypeWrapper.
    """
    def __init__(self, prompt_type: str, default_versions: dict[str, PromptFunction]):
        self.prompt_type = prompt_type
        self.default_versions = default_versions
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def __getattr__(self, version: str):
        def dynamic_wrapper(context: dict[str, Any]) -> list[Message]:
            source_description = context.get('source_description', 'default')
            
            # Dynamic selection for extract_nodes
            if self.prompt_type == 'extract_nodes' and 'source_description' in context:
                versions = get_extract_node_prompts(context)
                func = versions.get(version)
                if func is None:
                    func = self.default_versions.get(version)
                    if func is None:
                        raise AttributeError(f"Version '{version}' not found")
                    self.logger.info(f"🔍 [NODES] Using DEFAULT prompt '{version}' for source_description: '{source_description}' (no domain-specific prompt found)")
                else:
                    # Get more detailed information about the prompt being used
                    prompt_info = self._get_prompt_info(func, source_description)
                    self.logger.info(f"🎯 [NODES] Using DOMAIN-SPECIFIC prompt '{version}' for '{source_description}' → {prompt_info}")
            # Dynamic selection for extract_edges
            elif self.prompt_type == 'extract_edges' and 'source_description' in context:
                versions = get_extract_edge_prompts(context)
                func = versions.get(version)
                if func is None:
                    func = self.default_versions.get(version)
                    if func is None:
                        raise AttributeError(f"Version '{version}' not found")
                    self.logger.info(f"🔍 [EDGES] Using DEFAULT prompt '{version}' for source_description: '{source_description}' (no domain-specific prompt found)")
                else:
                    # Get more detailed information about the prompt being used
                    prompt_info = self._get_prompt_info(func, source_description)
                    self.logger.info(f"🎯 [EDGES] Using DOMAIN-SPECIFIC prompt '{version}' for '{source_description}' → {prompt_info}")
            else:
                func = self.default_versions.get(version)
                if func is None:
                    raise AttributeError(f"Version '{version}' not found")
                self.logger.info(f"🔍 [{self.prompt_type.upper()}] Using DEFAULT prompt '{version}' (no source_description in context)")
            
            # Apply the same wrapper logic as before
            messages = func(context)
            for message in messages:
                message.content += DO_NOT_ESCAPE_UNICODE if message.role == 'system' else ''
            return messages
        
        return dynamic_wrapper
    
    def _get_prompt_info(self, func: PromptFunction, source_description: str) -> str:
        """
        Extract information about which prompt file is being used.
        """
        try:
            # Get the function's module name to identify the prompt file
            module_name = func.__module__
            
            # Map source descriptions to human-readable domain names
            domain_mapping = {
                'aws_resources': 'AWS Cloud',
                'azure_resources': 'Azure Cloud', 
                'gcp_resources': 'GCP Cloud',
                'cloud_resources': 'Generic Cloud',
                'github_repo': 'GitHub Repository',
                'github_resources': 'GitHub Development',
                'cicd_resources': 'CI/CD Pipeline',
                'pipeline_resources': 'Deployment Pipeline',
                'logs_resources': 'Observability Logs',
                'metrics_resources': 'Observability Metrics',
                'traces_resources': 'Observability Traces',
                'observability_resources': 'General Observability',
                'monitoring_resources': 'Monitoring Systems'
            }
            
            domain_name = domain_mapping.get(source_description, source_description)
            
            # Extract the file path from module name
            if 'cloud.aws' in module_name:
                return f"cloud/aws/extract_{self.prompt_type}.py ({domain_name})"
            elif 'cloud.azure' in module_name:
                return f"cloud/azure/extract_{self.prompt_type}.py ({domain_name})"
            elif 'cloud.gcp' in module_name:
                return f"cloud/gcp/extract_{self.prompt_type}.py ({domain_name})"
            elif 'github' in module_name:
                return f"github/extract_{self.prompt_type}.py ({domain_name})"
            elif 'cicd' in module_name:
                return f"cicd/extract_{self.prompt_type}.py ({domain_name})"
            elif 'observability.logs' in module_name:
                return f"observability/logs/extract_{self.prompt_type}.py ({domain_name})"
            elif 'observability.metrics' in module_name:
                return f"observability/metrics/extract_{self.prompt_type}.py ({domain_name})"
            elif 'observability.traces' in module_name:
                return f"observability/traces/extract_{self.prompt_type}.py ({domain_name})"
            else:
                return f"{module_name} ({domain_name})"
                
        except Exception:
            return f"unknown prompt file ({source_description})"


class PromptLibraryWrapper:
    def __init__(self, library: PromptLibraryImpl):
        for prompt_type, versions in library.items():
            if prompt_type in ['extract_nodes', 'extract_edges']:
                # Use dynamic wrapper for both extract_nodes and extract_edges
                setattr(self, prompt_type, DynamicPromptTypeWrapper(prompt_type, versions))  # type: ignore[arg-type]
            else:
                # Use original wrapper for others
                setattr(self, prompt_type, PromptTypeWrapper(versions))  # type: ignore[arg-type]


PROMPT_LIBRARY_IMPL: PromptLibraryImpl = {
    'extract_nodes': extract_nodes_versions,
    'dedupe_nodes': dedupe_nodes_versions,
    'extract_edges': extract_edges_versions,
    'dedupe_edges': dedupe_edges_versions,
    'invalidate_edges': invalidate_edges_versions,
    'extract_edge_dates': extract_edge_dates_versions,
    'summarize_nodes': summarize_nodes_versions,
    'eval': eval_versions,
}
prompt_library: PromptLibrary = PromptLibraryWrapper(PROMPT_LIBRARY_IMPL)  # type: ignore[assignment]