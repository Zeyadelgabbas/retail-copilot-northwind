import dspy
from typing import Literal


class RouteQuestion(dspy.Signature):
    """Classify if question needs: 'rag' (docs only), 'sql' (database only), or 'hybrid' (both)
    
    Rules:
    - Use 'rag' if question asks about policies, return windows, definitions from documents
    - Use 'sql' if question asks for numbers, rankings, aggregations from database only
    - Use 'hybrid' if question needs both document context (dates, KPIs, definitions) AND database calculations
    
    Examples:
    - "What is the return window for Beverages?" → rag (policy from docs)
    - "Top 3 products by revenue" → sql (pure database query)
    - "Revenue during Summer Beverages 1997" → hybrid (dates from docs, revenue from DB)
    """
    question = dspy.InputField(desc="User question")
    route: Literal['rag', 'sql', 'hybrid'] = dspy.OutputField(desc="One word: rag, sql, or hybrid")


class GenerateSQL(dspy.Signature):
    """Generate SQLite query from natural language.
     
      - Prefer Orders + "Order Details" + Products joins.
      - Revenue: SUM(UnitPrice * Quantity * (1 - Discount)) from "orderitems". 
      - If needed, map categoryname via Categories join through Products.CategoryID.

      DONT use any column , table not mentioned in the database schema.
     
    """
    question = dspy.InputField(desc="User question")
    db_schema = dspy.InputField(desc="Database schema with tables and columns")
    context = dspy.InputField(desc="Document context with dates/definitions")
    sql = dspy.OutputField(desc="Valid SQLite query ONLY")


class RepairSQL(dspy.Signature):
    """Fix SQL query that produced an error.
    
    Table aliases:
    - o = Orders
    - oi = orderitems
    - p = Products
    - c = Categories
    - cu = Customers
    """
    question = dspy.InputField(desc="Original question")
    failed_sql = dspy.InputField(desc="SQL that failed")
    error = dspy.InputField(desc="Error message")
    db_schema = dspy.InputField(desc="Database schema")
    repaired_sql = dspy.OutputField(desc="Corrected SQLite query")


class SynthesizeAnswer(dspy.Signature):
    """Synthesize final answer from SQL results and/or documents.
    
    OUTPUT RULES:
    - For format_hint='int': return ONLY the number (e.g., "14")
    - For format_hint='float': return ONLY number with 2 decimals (e.g., "1234.56")
    - For format_hint='{...}': return valid JSON object (e.g., {"category": "Beverages", "quantity": 123})
    - For format_hint='List[{...}]': return valid JSON array (e.g., [{"product": "X", "revenue": 100.00}])
    - NO extra text, NO explanations in the answer field
    """
    question = dspy.InputField(desc="User question")
    format_hint = dspy.InputField(desc="Expected output format")
    sql_results = dspy.InputField(desc="SQL query results as JSON")
    doc_context = dspy.InputField(desc="Retrieved document chunks")
    answer = dspy.OutputField(desc="Final answer matching format_hint EXACTLY")
    explanation = dspy.OutputField(desc="One sentence (under 20 words)")


class RouterModule(dspy.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.classify = dspy.ChainOfThought(RouteQuestion)
    
    def forward(self, question, **kwargs):
        result = self.classify(question=question)
        route = result.route.lower().strip()
        if route not in ['rag', 'sql', 'hybrid']:
            route = 'hybrid'
        return dspy.Prediction(route=route)


class SQLGeneratorModule(dspy.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.generate = dspy.ChainOfThought(GenerateSQL)
    
    def forward(self, question, db_schema, context="", **kwargs):
        result = self.generate(
            question=question,
            db_schema=db_schema,
            context=context
        )
        return dspy.Prediction(sql=result.sql)


class SQLRepairModule(dspy.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.repair = dspy.ChainOfThought(RepairSQL)
    
    def forward(self, question, failed_sql, error, db_schema, **kwargs):
        result = self.repair(
            question=question,
            failed_sql=failed_sql,
            error=error,
            db_schema=db_schema
        )
        return dspy.Prediction(repaired_sql=result.repaired_sql)


class SynthesizerModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.synthesize = dspy.ChainOfThought(SynthesizeAnswer)
    
    def forward(self, question, format_hint, sql_results="", doc_context=""):
        result = self.synthesize(
            question=question,
            format_hint=format_hint,
            sql_results=sql_results,
            doc_context=doc_context
        )
        return dspy.Prediction(
            answer=result.answer,
            explanation=result.explanation
        )