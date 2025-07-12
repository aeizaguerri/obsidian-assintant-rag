"""
Text processing utilities for cleaning markdown content.

This module provides functions to clean and preprocess markdown text
for embedding generation by removing formatting elements while preserving
the actual content.
"""

import re
import os


def remove_frontmatter(content: str) -> str:
    """Remove YAML frontmatter from markdown content."""
    return re.sub(r'^---\s*\n.*?\n---\s*\n', '', content, flags=re.DOTALL)


def remove_code_blocks(content: str) -> str:
    """Remove code blocks and inline code from markdown content."""
    content = re.sub(r'```.*?```', '', content, flags=re.DOTALL)
    content = re.sub(r'`[^`]*`', '', content)
    return content


def remove_headers(content: str) -> str:
    """Remove markdown headers (# ## ### etc) while preserving text."""
    return re.sub(r'^#{1,6}\s+', '', content, flags=re.MULTILINE)


def process_obsidian_wikilinks(content: str) -> str:
    """Process Obsidian wikilinks [[page]] and [[page|display]] to preserve link information."""
    # Handle wikilinks with custom display text [[page|display]] -> display (from page)
    content = re.sub(r'\[\[([^\|\]]+)\|([^\]]+)\]\]', r'\2 (from \1)', content)
    # Handle simple wikilinks [[page]] -> page
    content = re.sub(r'\[\[([^\]]+)\]\]', r'\1', content)
    return content


def extract_obsidian_tags(content: str) -> list[str]:
    """Extract Obsidian tags (#tag) from content."""
    tags = re.findall(r'#([a-zA-Z0-9_-]+)', content)
    return tags


def extract_note_title(content: str, filename: str) -> str:
    """Extract note title from content or filename."""
    # Look for first H1 header
    h1_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
    if h1_match:
        return h1_match.group(1).strip()
    
    # Fall back to filename without extension
    return os.path.splitext(os.path.basename(filename))[0]


def get_obsidian_metadata(content: str, filepath: str) -> dict:
    """Extract comprehensive metadata from Obsidian note."""
    return {
        'title': extract_note_title(content, filepath),
        'tags': extract_obsidian_tags(content),
        'filepath': filepath,
        'filename': os.path.basename(filepath),
        'word_count': len(content.split()),
        'has_wikilinks': '[[' in content,
        'has_tags': '#' in content
    }


def process_links_and_images(content: str) -> str:
    """Remove images and convert links to plain text, handling Obsidian wikilinks specially."""
    # Process Obsidian wikilinks first
    content = process_obsidian_wikilinks(content)
    # Remove images ![alt](url)
    content = re.sub(r'!\[[^\]]*\]\([^\)]*\)', '', content)
    # Convert regular markdown links to plain text [text](url) -> text
    content = re.sub(r'\[([^\]]*)\]\([^\)]*\)', r'\1', content)
    return content


def remove_text_formatting(content: str) -> str:
    """Remove bold, italic, and strikethrough formatting."""
    # Bold and italic
    content = re.sub(r'\*\*([^\*]*)\*\*', r'\1', content)  # **bold**
    content = re.sub(r'\*([^\*]*)\*', r'\1', content)      # *italic*
    content = re.sub(r'__([^_]*)__', r'\1', content)       # __bold__
    content = re.sub(r'_([^_]*)_', r'\1', content)         # _italic_
    # Strikethrough
    content = re.sub(r'~~([^~]*)~~', r'\1', content)
    return content


def remove_structural_elements(content: str) -> str:
    """Remove horizontal rules, blockquotes, and list markers."""
    # Horizontal rules
    content = re.sub(r'^---+\s*$', '', content, flags=re.MULTILINE)
    content = re.sub(r'^\*\*\*+\s*$', '', content, flags=re.MULTILINE)
    # Blockquotes
    content = re.sub(r'^>\s*', '', content, flags=re.MULTILINE)
    # List markers
    content = re.sub(r'^\s*[-\*\+]\s+', '', content, flags=re.MULTILINE)
    content = re.sub(r'^\s*\d+\.\s+', '', content, flags=re.MULTILINE)
    return content


def remove_tables_and_html(content: str) -> str:
    """Remove table formatting and HTML tags."""
    # Table formatting
    content = re.sub(r'\|', ' ', content)
    content = re.sub(r'^[-\s\|:]+$', '', content, flags=re.MULTILINE)
    # HTML tags
    content = re.sub(r'<[^>]+>', '', content)
    return content


def normalize_whitespace(content: str) -> str:
    """Clean up extra whitespace and normalize spacing."""
    content = re.sub(r'\n\s*\n', '\n\n', content)  # Multiple newlines to double
    content = re.sub(r'[ \t]+', ' ', content)      # Multiple spaces to single
    return content.strip()


def clean_markdown_content(content: str) -> str:
    """
    Clean markdown content by removing all formatting elements.
    
    This function processes markdown content through several stages:
    1. Remove YAML frontmatter
    2. Remove code blocks and inline code
    3. Remove headers while preserving text
    4. Process links and images
    5. Remove text formatting (bold, italic, strikethrough)
    6. Remove structural elements (lists, blockquotes, horizontal rules)
    7. Remove tables and HTML
    8. Normalize whitespace
    
    Args:
        content: Raw markdown content as string
        
    Returns:
        Cleaned plain text suitable for embedding generation
    """
    cleaning_pipeline = [
        remove_frontmatter,
        remove_code_blocks,
        remove_headers,
        process_links_and_images,
        remove_text_formatting,
        remove_structural_elements,
        remove_tables_and_html,
        normalize_whitespace
    ]
    
    for clean_func in cleaning_pipeline:
        content = clean_func(content)
    
    return content