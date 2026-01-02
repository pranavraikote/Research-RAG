import logging


def setup_logging(log_level="INFO", log_file=None):
    """
    Set up logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Path to log file (optional)
    """

    handlers = [logging.StreamHandler()]
    
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


def format_citation(metadata):
    """
    Format paper citation in academic style.
    
    Args:
        metadata: Paper metadata dictionary
        
    Returns:
        Formatted citation string
    """

    authors = metadata.get("authors", "Unknown")
    title = metadata.get("title", "Unknown")
    venue = metadata.get("venue", "")
    year = metadata.get("year", "")
    
    citation = f"{authors}. {title}"
    if venue:
        citation += f". {venue}"
    if year:
        citation += f" ({year})"
    
    return citation