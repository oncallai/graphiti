"""
Helper functions and constants for prompt generation
"""

# Unicode handling
DO_NOT_ESCAPE_UNICODE = '\nDo not escape unicode characters.\n'

# Domain-specific source descriptions for better prompt selection
CLOUD_SOURCE_DESCRIPTIONS = {
    'aws_resources': 'AWS cloud infrastructure and services',
    'azure_resources': 'Azure cloud infrastructure and services', 
    'gcp_resources': 'Google Cloud Platform infrastructure and services',
    'cloud_resources': 'Generic cloud infrastructure and services'
}

GITHUB_SOURCE_DESCRIPTIONS = {
    'github_repo': 'GitHub repository and development resources',
    'github_resources': 'GitHub repositories, workflows, and development tools'
}

CICD_SOURCE_DESCRIPTIONS = {
    'cicd_resources': 'CI/CD pipelines and build systems',
    'pipeline_resources': 'Deployment pipelines and automation tools'
}

OBSERVABILITY_SOURCE_DESCRIPTIONS = {
    'logs_resources': 'Logging systems and log management',
    'metrics_resources': 'Metrics collection and monitoring systems',
    'traces_resources': 'Distributed tracing and trace analysis',
    'observability_resources': 'General observability and monitoring',
    'monitoring_resources': 'Monitoring and alerting systems'  # Legacy support
}

APPLICATION_SOURCE_DESCRIPTIONS = {
    'application_resources': 'Cross-domain application hubs and relationships'
}

# Combined source descriptions for easy lookup
ALL_SOURCE_DESCRIPTIONS = {
    **CLOUD_SOURCE_DESCRIPTIONS,
    **GITHUB_SOURCE_DESCRIPTIONS,
    **CICD_SOURCE_DESCRIPTIONS,
    **OBSERVABILITY_SOURCE_DESCRIPTIONS,
    **APPLICATION_SOURCE_DESCRIPTIONS
}

def get_domain_from_source_description(source_description: str) -> str:
    """
    Extract the domain from a source description.
    
    Args:
        source_description: The source description to analyze
        
    Returns:
        The main domain (cloud, github, cicd, observability, application, generic)
    """
    if source_description in CLOUD_SOURCE_DESCRIPTIONS:
        return 'cloud'
    elif source_description in GITHUB_SOURCE_DESCRIPTIONS:
        return 'github'
    elif source_description in CICD_SOURCE_DESCRIPTIONS:
        return 'cicd'
    elif source_description in OBSERVABILITY_SOURCE_DESCRIPTIONS:
        return 'observability'
    elif source_description in APPLICATION_SOURCE_DESCRIPTIONS:
        return 'application'
    else:
        return 'generic'

def get_subdomain_from_source_description(source_description: str) -> str:
    """
    Extract the subdomain from a source description.
    
    Args:
        source_description: The source description to analyze
        
    Returns:
        The subdomain (aws, azure, gcp, logs, metrics, traces, application, generic)
    """
    if source_description in ['aws_resources']:
        return 'aws'
    elif source_description in ['azure_resources']:
        return 'azure'
    elif source_description in ['gcp_resources']:
        return 'gcp'
    elif source_description in ['logs_resources']:
        return 'logs'
    elif source_description in ['metrics_resources']:
        return 'metrics'
    elif source_description in ['traces_resources']:
        return 'traces'
    elif source_description in ['application_resources']:
        return 'application'
    else:
        return 'generic'

def format_custom_prompt(domain: str, subdomain: str = None) -> str:
    """
    Format a custom prompt based on domain and subdomain.
    
    Args:
        domain: The main domain (cloud, github, cicd, observability)
        subdomain: The subdomain (aws, azure, gcp, logs, metrics, traces)
        
    Returns:
        A formatted custom prompt string
    """
    if domain == 'cloud' and subdomain:
        return f"\nFocus specifically on {subdomain.upper()} services and infrastructure components."
    elif domain == 'observability' and subdomain:
        return f"\nFocus specifically on {subdomain} collection and analysis."
    elif domain == 'github':
        return "\nFocus specifically on GitHub repositories, workflows, and development resources."
    elif domain == 'cicd':
        return "\nFocus specifically on CI/CD pipelines, build systems, and deployment automation."
    
    return ""

def validate_source_description(source_description: str) -> bool:
    """
    Validate if a source description is recognized.
    
    Args:
        source_description: The source description to validate
        
    Returns:
        True if the source description is recognized, False otherwise
    """
    return source_description in ALL_SOURCE_DESCRIPTIONS
