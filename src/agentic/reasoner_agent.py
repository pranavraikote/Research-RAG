"""
Reasoner Agent

Handles all reasoning tasks: analysis, comparison, and synthesis.
Simplified single agent that replaces Analyzer, Comparator, and Synthesizer.
"""

import logging
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


class ReasonerAgent(BaseAgent):
    
    def __init__(self, llm):
        """
        Initialize reasoner agent function.
        
        Args:
            llm: Language model for reasoning
        """
        
        super().__init__(
            name = "ReasonerAgent",
            description = "Performs reasoning tasks: analysis, comparison, and synthesis",
            llm = llm
        )
    
    def _determine_task_type(self, query, chunks):
        """
        Determine what type of reasoning task is needed function.
        
        Args:
            query: User query string
            chunks: Retrieved chunks
            
        Returns:
            Task type: 'analysis', 'comparison', 'synthesis', or 'general'
        """
        
        query_lower = query.lower()
        
        # Check for comparison keywords
        if any(word in query_lower for word in ["compare", "comparison", "versus", "vs", "difference", "similar", "contrast"]):
            return "comparison"
        
        # Check for gap detection/synthesis keywords
        if any(word in query_lower for word in ["gap", "missing", "unanswered", "future research", "limitation", "synthesize", "summary"]):
            return "synthesis"
        
        # Check for analysis keywords
        if any(word in query_lower for word in ["analyze", "extract", "findings", "claims", "methods", "results"]):
            return "analysis"
        
        # Default: general reasoning
        return "general"
    
    def _format_chunks(self, chunks, max_chunks = 10):
        """
        Format chunks for LLM input function.
        
        Args:
            chunks: List of paper chunks
            max_chunks: Maximum number of chunks to format
            
        Returns:
            Formatted string of chunks
        """
        
        formatted = []
        for i, chunk in enumerate(chunks[:max_chunks], 1):
            text = chunk.get("chunk", chunk.get("text", ""))
            title = chunk.get("metadata", {}).get("title", "Unknown")
            formatted.append(f"[Paper: {title}]\n{text[:500]}...\n")
        
        return "\n".join(formatted)
    
    def _group_chunks_by_paper(self, chunks):
        """
        Group chunks by paper title function.
        
        Args:
            chunks: List of chunks
            
        Returns:
            Dictionary mapping paper titles to chunks
        """
        
        papers_chunks = {}
        for chunk in chunks:
            paper_title = chunk.get("metadata", {}).get("title", "Unknown")
            if paper_title not in papers_chunks:
                papers_chunks[paper_title] = []
            papers_chunks[paper_title].append(chunk)
        
        return papers_chunks
    
    def _build_prompt(self, query, chunks, task_type):
        """
        Build appropriate prompt based on task type function.
        
        Args:
            query: User query
            chunks: Retrieved chunks
            task_type: Type of reasoning task
            
        Returns:
            Formatted prompt string
        """
        
        formatted_chunks = self._format_chunks(chunks)
        papers_chunks = self._group_chunks_by_paper(chunks)
        num_papers = len(papers_chunks)
        
        if task_type == "comparison" and num_papers >= 2:
            # Comparison prompt
            papers_text = []
            for i, (title, paper_chunks) in enumerate(papers_chunks.items(), 1):
                paper_text = self._format_chunks(paper_chunks[:3])  # Use first 3 chunks per paper
                papers_text.append(f"Paper {i} ({title}):\n{paper_text}\n")
            
            prompt = f"""Compare the following research papers based on the user's query.

User Query: {query}

Papers:
{''.join(papers_text)}

Provide a comprehensive comparison including:
1. Similarities in approaches, findings, or methods
2. Key differences
3. Which paper is stronger in which aspects
4. A direct answer to the user's query

Format as:
ANSWER:
[direct answer to the query]

SIMILARITIES:
- similarity 1
- similarity 2
...

DIFFERENCES:
- difference 1 (Paper 1 vs Paper 2)
- difference 2
...

STRENGTHS:
Paper 1: [strengths]
Paper 2: [strengths]

Comparison:"""
        
        elif task_type == "synthesis":
            # Synthesis/gap detection prompt
            prompt = f"""Analyze the following research paper chunks and provide a comprehensive answer to the user's query.

User Query: {query}

Paper chunks:
{formatted_chunks}

Provide:
1. A comprehensive answer to the query
2. Research gaps or unanswered questions identified
3. 2-3 follow-up research questions that would address these gaps

Format as:
ANSWER:
[comprehensive answer to the query]

GAPS:
- gap 1
- gap 2
...

FOLLOWUP QUESTIONS:
1. question 1
2. question 2
...

Synthesis:"""
        
        elif task_type == "analysis":
            # Analysis prompt
            prompt = f"""Analyze the following research paper chunks and extract key information.

User Query: {query}

Paper chunks:
{formatted_chunks}

Extract and provide:
1. Main claims and hypotheses (list 3-5 key claims)
2. Limitations and weaknesses mentioned
3. Methodology/approach used
4. Key experimental results or findings
5. A direct answer to the user's query

Format as:
ANSWER:
[direct answer to the query]

CLAIMS:
- claim 1
- claim 2
...

LIMITATIONS:
- limitation 1
- limitation 2
...

METHODOLOGY:
[description of approach]

RESULTS:
[key findings/results]

Analysis:"""
        
        else:
            # General reasoning prompt
            prompt = f"""Answer the following query based on the provided research paper chunks.

User Query: {query}

Paper chunks:
{formatted_chunks}

Provide a comprehensive answer that:
1. Directly addresses the query
2. Cites specific information from the papers
3. Synthesizes information across multiple papers if applicable

Answer:"""
        
        return prompt
    
    def _parse_response(self, content, task_type):
        """
        Parse LLM response based on task type function.
        
        Args:
            content: LLM response text
            task_type: Type of reasoning task
            
        Returns:
            Parsed results dictionary
        """
        
        result = {
            "agent": self.name,
            "task_type": task_type,
            "answer": "",
            "raw_response": content
        }
        
        # Extract answer section
        if "ANSWER:" in content:
            answer_section = content.split("ANSWER:")[1].split("\n\n")[0] if "\n\n" in content.split("ANSWER:")[1] else content.split("ANSWER:")[1].split("\n")[0]
            result["answer"] = answer_section.strip()
        else:
            # Fallback: use first paragraph
            result["answer"] = content.split("\n\n")[0][:500] if "\n\n" in content else content[:500]
        
        # Parse based on task type
        if task_type == "comparison":
            similarities = []
            differences = []
            strengths = {}
            current_section = None
            
            for line in content.split("\n"):
                line = line.strip()
                if line.startswith("SIMILARITIES:"):
                    current_section = "similarities"
                elif line.startswith("DIFFERENCES:"):
                    current_section = "differences"
                elif line.startswith("STRENGTHS:"):
                    current_section = "strengths"
                elif line and current_section:
                    if current_section == "similarities" and line.startswith("-"):
                        similarities.append(line[1:].strip())
                    elif current_section == "differences" and line.startswith("-"):
                        differences.append(line[1:].strip())
                    elif current_section == "strengths":
                        if "Paper 1:" in line:
                            strengths["paper1"] = line.split("Paper 1:")[-1].strip()
                        elif "Paper 2:" in line:
                            strengths["paper2"] = line.split("Paper 2:")[-1].strip()
            
            result["similarities"] = similarities
            result["differences"] = differences
            result["strengths"] = strengths
        
        elif task_type == "synthesis":
            gaps = []
            followup_questions = []
            current_section = None
            
            for line in content.split("\n"):
                line = line.strip()
                if line.startswith("GAPS:"):
                    current_section = "gaps"
                elif line.startswith("FOLLOWUP QUESTIONS:"):
                    current_section = "followup"
                elif line and current_section:
                    if current_section == "gaps" and line.startswith("-"):
                        gaps.append(line[1:].strip())
                    elif current_section == "followup" and (line[0].isdigit() or line.startswith("-")):
                        question = line.split(".", 1)[-1].strip() if "." in line else line[1:].strip()
                        if question:
                            followup_questions.append(question)
            
            result["gaps"] = gaps
            result["followup_questions"] = followup_questions[:3]  # Limit to 3
        
        elif task_type == "analysis":
            claims = []
            limitations = []
            methods = ""
            results = ""
            current_section = None
            
            for line in content.split("\n"):
                line = line.strip()
                if line.startswith("CLAIMS:"):
                    current_section = "claims"
                elif line.startswith("LIMITATIONS:"):
                    current_section = "limitations"
                elif line.startswith("METHODOLOGY:"):
                    current_section = "methods"
                elif line.startswith("RESULTS:"):
                    current_section = "results"
                elif line and current_section:
                    if current_section == "claims" and line.startswith("-"):
                        claims.append(line[1:].strip())
                    elif current_section == "limitations" and line.startswith("-"):
                        limitations.append(line[1:].strip())
                    elif current_section == "methods":
                        methods += line + " "
                    elif current_section == "results":
                        results += line + " "
            
            result["claims"] = claims
            result["limitations"] = limitations
            result["methods"] = methods.strip()
            result["results"] = results.strip()
        
        return result
    
    def execute(self, task, context):
        """
        Execute reasoning task function.
        
        Args:
            task: User query string
            context: Context with chunks and query
            
        Returns:
            Reasoning results
        """
        
        query = task
        chunks = context.get("chunks", [])
        
        if not chunks:
            logger.warning("No chunks provided for reasoning")
            return {
                "agent": self.name,
                "task_type": "general",
                "answer": "No relevant papers found to answer the query.",
                "gaps": [],
                "followup_questions": []
            }
        
        # Determine task type
        task_type = self._determine_task_type(query, chunks)
        logger.debug(f"Determined task type: {task_type} for query: {query[:50]}...")
        
        # Build prompt
        prompt = self._build_prompt(query, chunks, task_type)
        
        # Show prompt if requested
        import os
        if os.getenv('AGENTIC_SHOW_PROMPTS'):
            logger.info(f"\n{'='*60}\nREASONER PROMPT ({task_type}):\n{'='*60}\n{prompt}\n{'='*60}")
        
        # Use streaming to avoid tokenizer overflow issues (same as standalone system)
        content_parts = []
        try:
            stream_iter = self.llm.stream(prompt)
            for chunk in stream_iter:
                text = chunk.text if hasattr(chunk, 'text') else str(chunk)
                content_parts.append(text)
                # Stop if we've collected enough (safety limit)
                if len("".join(content_parts)) > 2000:
                    logger.debug("Reached safety limit, stopping stream")
                    break
            content = "".join(content_parts)
        except (OverflowError, RuntimeError, StopIteration, Exception) as e:
            # Catch any streaming error and use what we have
            content = "".join(content_parts) if content_parts else ""
            if not content:
                logger.warning(f"Streaming failed: {e}, using invoke as fallback")
                try:
                    response = self.llm.invoke(prompt)
                    content = response.content if hasattr(response, 'content') else str(response)
                except Exception as e2:
                    logger.error(f"Invoke also failed: {e2}, returning empty")
                    content = "Reasoning completed but generation failed."
        
        logger.debug(f"LLM response length: {len(content)} characters")
        
        # Parse response
        result = self._parse_response(content, task_type)
        result["chunks_used"] = len(chunks)
        
        logger.info(f"Reasoning completed: task_type={task_type}, answer_length={len(result['answer'])}")
        
        return result
