"""
Metadata extraction utilities for Obsidian documents.

This module provides enhanced metadata extraction functionality
without circular dependencies.
"""

import re
import os
from typing import Dict, List
from markdown_processing import get_obsidian_metadata

def extract_enhanced_metadata(content: str, file_path: str) -> Dict:
    """Extract enhanced metadata including backlinks and document structure"""
    base_metadata = get_obsidian_metadata(content, file_path)
    
    # Extract backlinks (mentions of other notes)
    backlink_pattern = r'\[\[([^\|\]]+)(?:\|([^\]]+))?\]\]'
    backlinks = []
    for match in re.finditer(backlink_pattern, content):
        target_note = match.group(1)
        display_text = match.group(2) if match.group(2) else target_note
        backlinks.append({'target': target_note, 'display': display_text})
    
    # Extract forward links [text](url)
    forward_links = []
    link_pattern = r'\[([^\]]+)\]\(([^\)]+)\)'
    for match in re.finditer(link_pattern, content):
        forward_links.append({'text': match.group(1), 'url': match.group(2)})
    
    # Extract aliases (alternative names for the note)
    aliases = []
    alias_pattern = r'(?:^|\n)aliases?:\s*(.+?)(?:\n|$)'
    alias_match = re.search(alias_pattern, content, re.IGNORECASE)
    if alias_match:
        aliases = [alias.strip() for alias in alias_match.group(1).split(',')]
    
    # Count headers by level
    header_counts = {}
    for i in range(1, 7):
        header_pattern = f'^{"#" * i}\\s+'
        header_counts[f'h{i}'] = len(re.findall(header_pattern, content, re.MULTILINE))
    
    # Extract code blocks count
    code_blocks = len(re.findall(r'```[\s\S]*?```', content))
    
    # Calculate readability metrics
    sentences = len(re.findall(r'[.!?]+', content))
    words = len(content.split())
    avg_words_per_sentence = words / sentences if sentences > 0 else 0
    
    enhanced_metadata = {
        **base_metadata,
        'backlinks': backlinks,
        'forward_links': forward_links,
        'aliases': aliases,
        'header_counts': header_counts,
        'code_blocks': code_blocks,
        'sentence_count': sentences,
        'avg_words_per_sentence': avg_words_per_sentence,
        'note_complexity': len(backlinks) + len(forward_links) + sum(header_counts.values())
    }
    
    return enhanced_metadata