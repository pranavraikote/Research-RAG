"""
Agentic RAG orchestrator — coordinates the v1 two-agent workflow (Retriever + Reasoner).
"""

import logging
import time
from typing import Any, Dict, Generator, List, Optional

from .retriever_agent import RetrieverAgent
from .reasoner_agent import ReasonerAgent

logger = logging.getLogger(__name__)


class AgenticRAGOrchestrator:

    def __init__(self, rag_chain, llm):
        """
        Args:
            rag_chain: RAGChain instance for retrieval.
            llm: Language model for all agents.
        """
        self.rag_chain = rag_chain
        self.llm = llm
        self.retriever_agent = RetrieverAgent(rag_chain, llm)
        self.reasoner_agent = ReasonerAgent(llm)
        self.execution_history: List[Dict[str, Any]] = []

    def _determine_workflow(self, query: str) -> str:
        """Return workflow name based on keywords in the query."""
        q = query.lower()
        if any(w in q for w in ["compare", "comparison", "versus", "vs", "difference", "similar", "contrast"]):
            return "comparison"
        if any(w in q for w in ["gap", "missing", "unanswered", "future research", "limitation", "synthesize"]):
            return "synthesis"
        return "general"

    def query(
        self,
        question: str,
        top_k: int = 5,
        filters: Optional[Dict] = None,
        workflow: Optional[str] = None,
        max_iterations: int = 3,
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Execute an agentic query and yield progress + final result dicts.

        Args:
            question: User query.
            top_k: Number of papers to retrieve per iteration.
            filters: Metadata filters (conference, year, title).
            workflow: Workflow pattern; auto-determined if None.
            max_iterations: Maximum retrieval-reasoning iterations.

        Yields:
            Dicts with a 'type' key: workflow, iteration, agent, retrieval,
            reasoning, refinement, complete, or error.
        """
        start_time = time.time()
        context: Dict[str, Any] = {
            "query": question,
            "top_k": top_k,
            "filters": filters,
            "retrieval": {},
            "reasoning": {},
        }

        if workflow is None:
            workflow = self._determine_workflow(question)
            logger.info("Auto-determined workflow: %s  query: %s...", workflow, question[:50])
        else:
            logger.info("Using specified workflow: %s", workflow)

        yield {"type": "workflow", "workflow": workflow, "message": f"Using {workflow} workflow pattern"}

        for iteration in range(1, max_iterations + 1):
            yield {"type": "iteration", "iteration": iteration, "message": f"Starting iteration {iteration}"}

            # Retrieval
            logger.info("[Iteration %d] Retrieval phase", iteration)
            yield {"type": "agent", "agent": "RetrieverAgent", "message": "Retrieving relevant papers..."}

            retrieval_start = time.time()
            retrieval_result = self.retriever_agent.execute(task=question, context=context)
            logger.info("[Iteration %d] Retrieval: %d chunks in %.2fs",
                        iteration, retrieval_result.get("count", 0), time.time() - retrieval_start)
            context["retrieval"] = retrieval_result

            if retrieval_result.get("count", 0) == 0:
                logger.warning("[Iteration %d] No papers retrieved", iteration)
                yield {"type": "error", "message": "No papers retrieved. Try a different query."}
                return

            yield {"type": "retrieval", "results": retrieval_result, "chunks_count": retrieval_result["count"]}

            # Reasoning
            logger.info("[Iteration %d] Reasoning phase", iteration)
            yield {"type": "agent", "agent": "ReasonerAgent", "message": "Reasoning over papers..."}

            reasoning_start = time.time()
            reasoning_result = self.reasoner_agent.execute(
                task=question,
                context={"chunks": retrieval_result.get("chunks", []), "query": question},
            )
            logger.info("[Iteration %d] Reasoning: task_type=%s  gaps=%d  %.2fs",
                        iteration, reasoning_result.get("task_type", "general"),
                        len(reasoning_result.get("gaps", [])),
                        time.time() - reasoning_start)
            context["reasoning"] = reasoning_result

            yield {"type": "reasoning", "results": reasoning_result, "task_type": reasoning_result.get("task_type", "general")}

            # Refine if gaps remain and iterations are available
            followup = reasoning_result.get("followup_questions", [])
            if reasoning_result.get("gaps") and iteration < max_iterations and followup:
                question = followup[0]
                yield {"type": "refinement", "message": f"Gap detected. Refining query.", "new_query": question}
                continue

            break

        total_time = time.time() - start_time
        reasoning_result = context.get("reasoning", {})
        logger.info("Query completed: %d iteration(s) in %.2fs", iteration, total_time)

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
            "task_type": reasoning_result.get("task_type", "general"),
        }

        self.execution_history.append({
            "question": question,
            "workflow": workflow,
            "iterations": iteration,
            "total_time": total_time,
        })
