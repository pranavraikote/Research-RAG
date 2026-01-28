"""
Agentic RAG Orchestrator

Coordinates simplified two-agent workflow: Retriever + Reasoner.
"""

import time
import logging
from typing import Dict, List, Any, Optional, Generator

from .retriever_agent import RetrieverAgent
from .reasoner_agent import ReasonerAgent

logger = logging.getLogger(__name__)


class AgenticRAGOrchestrator:
    
    def __init__(self, rag_chain, llm):
        """
        Initialize orchestrator with agents function.
        
        Args:
            rag_chain: RAGChain instance for retrieval
            llm: Language model for all agents
        """
        
        self.rag_chain = rag_chain
        self.llm = llm
        
        # Initialize simplified two-agent system
        self.retriever_agent = RetrieverAgent(rag_chain, llm)
        self.reasoner_agent = ReasonerAgent(llm)
        
        self.execution_history = []
    
    def _determine_workflow(self, query, context):
        """
        Determine which workflow pattern to use based on query function.
        
        Args:
            query: User query string
            context: Context dictionary
            
        Returns:
            Workflow pattern name (simplified: comparison, synthesis, or general)
        """
        
        query_lower = query.lower()
        
        # Check for comparison keywords
        if any(word in query_lower for word in ["compare", "comparison", "versus", "vs", "difference", "similar", "contrast"]):
            return "comparison"
        
        # Check for gap detection/synthesis keywords
        if any(word in query_lower for word in ["gap", "missing", "unanswered", "future research", "limitation", "synthesize"]):
            return "synthesis"
        
        # Default: general reasoning
        return "general"
    
    def query(self, question, top_k = 5, filters = None, workflow = None, max_iterations = 3):
        """
        Execute agentic query with multi-agent workflow function.
        
        Args:
            question: User query
            top_k: Number of papers to retrieve
            filters: Metadata filters
            workflow: Workflow pattern (auto-determined if None)
            max_iterations: Maximum iterations for iterative refinement
            
        Yields:
            Progress updates and final results
        """
        start_time = time.time()
        context = {
            "query": question,
            "top_k": top_k,
            "filters": filters,
            "retrieval": {},
            "reasoning": {}
        }
        
        # Determine workflow
        if workflow is None:
            workflow = self._determine_workflow(question, context)
            logger.info(f"Auto-determined workflow: {workflow} for query: {question[:50]}...")
        else:
            logger.info(f"Using specified workflow: {workflow}")
        
        yield {
            "type": "workflow",
            "workflow": workflow,
            "message": f"Using {workflow} workflow pattern"
        }
        
        iteration = 0
        while iteration < max_iterations:
            iteration += 1
            
            yield {
                "type": "iteration",
                "iteration": iteration,
                "message": f"Starting iteration {iteration}"
            }
            
            # Step 1: Retrieval
            logger.info(f"[Iteration {iteration}] Starting retrieval phase")
            yield {
                "type": "agent",
                "agent": "RetrieverAgent",
                "message": "Retrieving relevant papers..."
            }
            
            retrieval_start = time.time()
            retrieval_result = self.retriever_agent.execute(
                task=question,
                context=context
            )
            retrieval_time = time.time() - retrieval_start
            context["retrieval"] = retrieval_result
            
            chunks_count = retrieval_result.get("count", 0)
            logger.info(f"[Iteration {iteration}] Retrieval completed: {chunks_count} chunks in {retrieval_time:.2f}s")
            
            yield {
                "type": "retrieval",
                "results": retrieval_result,
                "chunks_count": chunks_count
            }
            
            if chunks_count == 0:
                logger.warning(f"[Iteration {iteration}] No papers retrieved for query: {question}")
                yield {
                    "type": "error",
                    "message": "No papers retrieved. Try a different query."
                }
                return
            
            # Step 2: Reasoning (replaces Analysis + Comparison + Synthesis)
            logger.info(f"[Iteration {iteration}] Starting reasoning phase")
            yield {
                "type": "agent",
                "agent": "ReasonerAgent",
                "message": "Reasoning over papers..."
            }
            
            chunks = retrieval_result.get("chunks", [])
            
            reasoning_start = time.time()
            reasoning_result = self.reasoner_agent.execute(
                task=question,
                context={"chunks": chunks, "query": question}
            )
            reasoning_time = time.time() - reasoning_start
            context["reasoning"] = reasoning_result
            
            task_type = reasoning_result.get("task_type", "general")
            gaps_count = len(reasoning_result.get("gaps", []))
            questions_count = len(reasoning_result.get("followup_questions", []))
            logger.info(f"[Iteration {iteration}] Reasoning completed in {reasoning_time:.2f}s (task_type={task_type}, gaps={gaps_count}, questions={questions_count})")
            
            yield {
                "type": "reasoning",
                "results": reasoning_result,
                "task_type": task_type
            }
            
            # Check if we need another iteration (gap detected that needs more retrieval)
            gaps = reasoning_result.get("gaps", [])
            if gaps and iteration < max_iterations:
                # Generate refined query based on gaps
                followup_questions = reasoning_result.get("followup_questions", [])
                if followup_questions:
                    question = followup_questions[0]  # Use first follow-up question
                    yield {
                        "type": "refinement",
                        "message": f"Gap detected. Refining query: {question}",
                        "new_query": question
                    }
                    continue
            
            # No more iterations needed
            break
        
        # Final result
        total_time = time.time() - start_time
        
        logger.info(f"Query completed: {iteration} iteration(s) in {total_time:.2f}s")
        logger.debug(f"Final context keys: {list(context.keys())}")
        
        reasoning_result = context.get("reasoning", {})
        
        yield {
            "type": "complete",
            "question": question,
            "workflow": workflow,
            "iterations": iteration,
            "total_time": total_time,
            "retrieval": context["retrieval"],
            "reasoning": reasoning_result,
            "answer": reasoning_result.get("answer", ""),
            "gaps": reasoning_result.get("gaps", []),
            "followup_questions": reasoning_result.get("followup_questions", []),
            "task_type": reasoning_result.get("task_type", "general")
        }
        
        # Store in history
        self.execution_history.append({
            "question": question,
            "workflow": workflow,
            "iterations": iteration,
            "total_time": total_time,
            "context": context
        })
        
        logger.debug(f"Execution history now has {len(self.execution_history)} entries")

