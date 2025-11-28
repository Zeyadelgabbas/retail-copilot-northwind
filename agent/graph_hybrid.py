from typing import TypedDict, List, Any
from langgraph.graph import StateGraph, END
import json
import re

from agent.dspy_signatures import (
    RouterModule, SQLGeneratorModule, SQLRepairModule, SynthesizerModule
)
from agent.rag.retreive import BM25Retriever
from agent.tools.sqlite_tool import SQLiteTool
from agent.logger import get_logger

logger = get_logger(__name__)


class AgentState(TypedDict):
    # Input
    id: str
    question: str
    format_hint: str
    
    # Routing
    route: str
    
    # RAG
    doc_chunks: List[dict]
    doc_context: str
    
    # SQL
    sql: str
    sql_results: str
    sql_error: str
    repair_count: int
    
    # Output
    final_answer: Any
    explanation: str
    confidence: float
    citations: List[str]
    
    # Trace
    trace: List[str]


class HybridAgent:
    def __init__(self, retriever: BM25Retriever, db_tool: SQLiteTool,
                 router: RouterModule, sql_gen: SQLGeneratorModule,
                 sql_repair: SQLRepairModule, synth: SynthesizerModule):
        self.retriever = retriever
        self.db_tool = db_tool
        self.router = router
        self.sql_gen = sql_gen
        self.sql_repair = sql_repair
        self.synth = synth
        self.db_schema_cache = db_tool.format_schema_for_llm()
        
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build LangGraph with routing, retrieval, SQL generation/repair, synthesis"""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("route", self.route_node)
        workflow.add_node("retrieve", self.retrieve_node)
        workflow.add_node("generate_sql", self.generate_sql_node)
        workflow.add_node("execute_sql", self.execute_sql_node)
        workflow.add_node("repair_sql", self.repair_sql_node)
        workflow.add_node("synthesize", self.synthesize_node)
        
        # Define edges
        workflow.set_entry_point("route")
        
        workflow.add_conditional_edges(
            "route",
            self.route_decision,
            {
                "rag": "retrieve",
                "sql": "generate_sql",
                "hybrid": "retrieve"
            }
        )
        
        workflow.add_conditional_edges(
            "retrieve",
            self.after_retrieve_decision,
            {
                "needs_sql": "generate_sql",
                "skip_sql": "synthesize"
            }
        )
        
        workflow.add_edge("generate_sql", "execute_sql")
        
        workflow.add_conditional_edges(
            "execute_sql",
            self.execute_decision,
            {
                "success": "synthesize",
                "repair": "repair_sql",
                "failed": "synthesize"
            }
        )
        
        workflow.add_edge("repair_sql", "execute_sql")
        workflow.add_edge("synthesize", END)
        
        return workflow.compile()
    
    def route_node(self, state: AgentState) -> AgentState:
        """Node 1: Route the question"""
        state["trace"].append("route: classifying question")
        result = self.router(question=state["question"])
        state["route"] = result.route
        state["trace"].append(f"route: selected {result.route}")
        logger.info(f"Question: {state['question']}")
        logger.info(f"Route: {state['route']}")
        return state
    
    def retrieve_node(self, state: AgentState) -> AgentState:
        """Node 2: Retrieve relevant documents"""
        state["trace"].append("retrieve: fetching doc chunks")
        chunks = self.retriever.retrieve(state["question"], top_k=3)
        logger.info(f"Retrieved documents : \n{[c.content for c in chunks]}")
        
        state["doc_chunks"] = [
            {"id": c.chunk_id, "content": c.content, "score": c.score}
            for c in chunks
        ]
        state["doc_context"] = self.retriever.get_context_string(chunks)
        state["trace"].append(f"retrieve: found {len(chunks)} chunks")
        
        # Add doc citations
        for chunk in chunks:
            if chunk.chunk_id not in state["citations"]:
                state["citations"].append(chunk.chunk_id)
        
        logger.info(f"Retrieved {len(chunks)} docs")
        return state
    
    def generate_sql_node(self, state: AgentState) -> AgentState:
        """Node 3: Generate SQL using DSPy"""
        state["trace"].append("generate_sql: creating query")
    
        result = self.sql_gen(
            question=state["question"],
            db_schema=self.db_schema_cache,
            context=state.get("doc_context", "")
            )
        
        # Clean SQL
        sql = result.sql.strip()
        sql = sql.replace("```sql", "").replace("```", "").strip()
        
        state["sql"] = sql
        state["trace"].append(f"generate_sql: {sql[:80]}...")
        logger.info(f"Generated SQL: {sql}")
        return state
    
    def execute_sql_node(self, state: AgentState) -> AgentState:
        """Node 4: Execute SQL"""
        state["trace"].append("execute_sql: running query")
        
        result = self.db_tool.execute_query(state["sql"])
        
        if result.success:
            logger.info(f" SQL RESULTS : {result.rows}")
            # Convert rows to JSON
            rows_as_dicts = []
            row_dict={}
            for row in result.rows:
                row_dict = {}
                for i, col in enumerate(result.columns):
                    row_dict[col] = row[i]
                rows_as_dicts.append(row_dict)
            
            state["sql_results"] = json.dumps(rows_as_dicts, default=str)
            state["sql_error"] = ""
            state["trace"].append(f"execute_sql: success - {len(result.rows)} rows")
            logger.info(f"SQL success: {len(result.rows)} rows")
            logger.info(f"row content : {row_dict}")
            
            # Add table citations
            for table in result.tables_used:
                if table not in state["citations"]:
                    state["citations"].append(table)
        else:
            state["sql_error"] = result.error
            state["trace"].append(f"execute_sql: failed - {result.error[:100]}")
            logger.error(f"SQL error: {result.error}")
        
        return state
    
    def repair_sql_node(self, state: AgentState) -> AgentState:
        """Node 5: Repair failed SQL"""
        state["repair_count"] += 1
        state["trace"].append(f"repair_sql: attempt {state['repair_count']}")
        
        result = self.sql_repair(
            question=state["question"],
            failed_sql=state["sql"],
            error=state["sql_error"],
            db_schema=self.db_schema_cache
        )
        
        # Clean repaired SQL
        sql = result.repaired_sql.strip()
        sql = sql.replace("```sql", "").replace("```", "").strip()
        
        state["sql"] = sql
        state["trace"].append(f"repair_sql: new query")
        logger.info(f"Repaired SQL: {sql}")
        return state
    
    def synthesize_node(self, state: AgentState) -> AgentState:
        """Node 6: Synthesize final answer"""
        state["trace"].append("synthesize: creating final answer")
        
        try:
            result = self.synth(
                question=state["question"],
                format_hint=state["format_hint"],
                sql_results=state.get("sql_results", ""),
                doc_context=state.get("doc_context", "")
            )
            
            # Parse answer
            logger.info(f"ANSWER FINAL === {result.answer}")
            logger.info(f"EXPLANATION === {result.explanation}")
            
            
            
            state["final_answer"] = result.answer
            state["explanation"] = result.explanation[:200]
            
        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            state["final_answer"] = self._fallback_answer(state)
            state["explanation"] = "Extracted from data"
        
        # Calculate confidence
        state["confidence"] = self._calculate_confidence(state)
        
        state["trace"].append("synthesize: complete")
        logger.info(f"Final answer: {state['final_answer']}")
        
        return state
    
    def _fallback_answer(self, state: AgentState):
        """Extract answer directly from SQL results or docs"""
        format_hint = state["format_hint"]
        
        # Try SQL results first
        if state.get("sql_results"):
            try:
                data = json.loads(state["sql_results"])
                
                if format_hint == "int":
                    if isinstance(data, list) and data:
                        first_val = list(data[0].values())[0]
                        return int(first_val)
                    return 0
                
                elif format_hint == "float":
                    if isinstance(data, list) and data:
                        first_val = list(data[0].values())[0]
                        return round(float(first_val), 2)
                    return 0.0
                
                elif "list[" in format_hint.lower():
                    return data if isinstance(data, list) else []
                
                elif "{" in format_hint:
                    return data[0] if isinstance(data, list) and data else {}
            except:
                pass
        
        # Try doc context for RAG questions
        if state.get("doc_context") and format_hint == "int":
            numbers = re.findall(r'\d+', state["doc_context"])
            if numbers:
                return int(numbers[0])
        
        # Ultimate fallback
        return self._get_default_value(format_hint)
    
    def _get_default_value(self, format_hint: str):
        """Get default value for format_hint"""
        if format_hint == "int":
            return 0
        elif format_hint == "float":
            return 0.0
        elif "list[" in format_hint.lower():
            return []
        elif "{" in format_hint:
            return {}
        return ""
    
    def _calculate_confidence(self, state: AgentState) -> float:
        """
        Calculate confidence using simple heuristic:
        
        Formula:
            confidence = base + retrieval_score + sql_score + data_score - repair_penalty
            Clamped to [0.1, 1.0]
        
        Components:
            - Base: 0.3 (starting point)
            - Retrieval: 0-0.3 (based on avg BM25 score)
            - SQL Success: 0.3 if executed without errors, else 0
            - Data Present: 0.2 if results non-empty, else 0
            - Repair Penalty: -0.1 per repair attempt
        """
        
        # base score
        score = 0.3

        #  Retrieval score: Average BM25 score normalized to 0-0.3 range
        if state.get("doc_chunks"):
            # Calculate average BM25 score across all retrieved chunks
            bm25_scores = [chunk["score"] for chunk in state["doc_chunks"]]
            avg_bm25 = sum(bm25_scores) / len(bm25_scores)
            
            # Divide by 10 and clip at 0.3
            retrieval_score = min(avg_bm25 / 10.0, 0.3)
            score += retrieval_score
        
        # SQL execution score: 0.3 if successful, 0 otherwise
        if state.get("route") in ["sql", "hybrid"]:
            if state.get("sql_results") and not state.get("sql_error"):
                score += 0.3
        
        # Data presence score: 0.2 if we have non-empty results
        if state.get("sql_results"):
            try:
                data = json.loads(state["sql_results"])
                if data:  # Non-empty list
                    score += 0.2
            except:
                pass 
        elif state.get("route") == "rag" and state.get("doc_chunks"):
            # For RAG-only questions, having docs adds score
            score += 0.2
        
        # Repair penalty: -0.1 for each repair attempt 
        repair_count = state.get("repair_count", 0)
        score -= 0.1 * repair_count
        
        #  clip to (0.1 , 1)
        return max(0.1, min(1.0, score))
    
    # Conditional edge functions
    def route_decision(self, state: AgentState) -> str:
        return state["route"]
    
    def after_retrieve_decision(self, state: AgentState) -> str:
        if state["route"] in ["sql", "hybrid"]:
            return "needs_sql"
        return "skip_sql"
    
    def execute_decision(self, state: AgentState) -> str:
        if state.get("sql_error"):
            if state["repair_count"] < 2:
                return "repair"
            else:
                return "failed"
        return "success"
    
    # Initializing
    def run(self, question: str, question_id: str = "", format_hint: str = "str") -> dict:
        """Run the agent on a question"""
        initial_state = AgentState(
            id=question_id,
            question=question,
            format_hint=format_hint,
            route="",
            doc_chunks=[],
            doc_context="",
            sql="",
            sql_results="",
            sql_error="",
            repair_count=0,
            final_answer=None,
            explanation="",
            confidence=0.0,
            citations=[],
            trace=[]
        )
        
        final_state = self.graph.invoke(initial_state)
        
        return {
            "id":final_state['id'],
            "final_answer": final_state["final_answer"],
            "sql": final_state.get("sql", ""),
            "confidence": final_state["confidence"],
            "explanation": final_state["explanation"],
            "citations": final_state["citations"],
        }