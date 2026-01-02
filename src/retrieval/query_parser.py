import re

class QueryParser:
    
    # Common conference names
    CONFERENCES = {
        'acl': 'ACL',
        'emnlp': 'EMNLP',
        'naacl': 'NAACL',
        'eacl': 'EACL',
        'coling': 'COLING',
        'iclr': 'ICLR',
        'neurips': 'NeurIPS',
        'nips': 'NeurIPS',
        'icml': 'ICML',
        'aaai': 'AAAI',
        'ijcai': 'IJCAI',
        'acl-long': 'ACL',
        'acl-short': 'ACL',
        'acl-demo': 'ACL'
    }
    
    def __init__(self):
        """
        Initialize query parser.
        """
        
        # Building regex pattern for conference detection
        conf_pattern = '|'.join(re.escape(conf.lower()) for conf in self.CONFERENCES.keys())
        self.conf_regex = re.compile(r'\b(' + conf_pattern + r')\b', re.IGNORECASE)
        
        # Year patterns: 4-digit years, ranges like "2020-2024", "2020 to 2024"
        self.year_regex = re.compile(r'\b(19|20)\d{2}\b')
        self.year_range_regex = re.compile(r'\b(19|20)\d{2}\s*[-–—]\s*(19|20)\d{2}\b')
        self.year_to_regex = re.compile(r'\b(19|20)\d{2}\s+to\s+(19|20)\d{2}\b', re.IGNORECASE)
    
    def parse(self, query):
        """
        Query parsing function.
        
        Args:
            query: Query string
        
        Returns:
            result: Tuple of (cleaned_query, filters_dict)
        """
        
        filters = {}
        original_query = query
        
        # Extracting conferences
        conferences = self._extract_conferences(query)
        if conferences:
            filters['conference'] = conferences[0] if len(conferences) == 1 else conferences
        
        # Extracting years
        year_filter = self._extract_years(query)
        if year_filter:
            filters['year'] = year_filter
        
        # Extracting title mentions (look for quoted text or "paper titled X" patterns)
        title_filter = self._extract_title(query)
        if title_filter:
            filters['title'] = title_filter
        
        return original_query, filters if filters else None
    
    def _extract_conferences(self, query):
        """
        Extracting conference names function.
        
        Args:
            query: Query string
        
        Returns:
            result: List of conference names or None
        """
        
        matches = self.conf_regex.findall(query.lower())
        if not matches:
            return None
        
        # Mapping to conference names
        conferences = []
        for match in matches:
            conf = self.CONFERENCES.get(match.lower(), match.upper())
            if conf not in conferences:
                conferences.append(conf)
        
        return conferences if conferences else None
    
    def _extract_years(self, query):
        """
        Extracting year function.
        
        Args:
            query: Query string
        
        Returns:
            result: Year filter
        """
        
        # Checking for year ranges first
        range_match = self.year_range_regex.search(query)
        if range_match:
            start_year, end_year = map(int, range_match.group().replace('–', '-').replace('—', '-').split('-'))
            return {'min': start_year, 'max': end_year}
        
        range_match = self.year_to_regex.search(query)
        if range_match:
            parts = range_match.group().split()
            start_year = int(parts[0])
            end_year = int(parts[-1])
            return {'min': start_year, 'max': end_year}
        
        # Checking for multiple years
        years = [int(m.group()) for m in self.year_regex.finditer(query)]
        if not years:
            return None
        
        # Removing duplicates and sorting
        years = sorted(list(set(years)))
        
        # If multiple years, returning as list
        if len(years) > 1:
            return years
        
        # Single year
        return years[0]
    
    def _extract_title(self, query):
        """
        Extracting paper title function.
        
        Args:
            query: Query string
        
        Returns:
            result: Title string or None
        """
        
        # Looking for quoted titles
        quoted = re.findall(r'"([^"]+)"', query)
        if quoted:
            return quoted[0] if len(quoted) == 1 else quoted
        
        # Looking for "paper titled X" or "titled X" patterns
        titled_match = re.search(r'(?:paper\s+)?titled\s+["\']?([^"\']+)["\']?', query, re.IGNORECASE)
        if titled_match:
            return titled_match.group(1).strip()
        
        return None

