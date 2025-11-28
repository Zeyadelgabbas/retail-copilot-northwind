import sqlite3
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class SQLResult:
    """Result of SQL query execution"""
    success: bool
    columns: List[str]
    rows: List[Tuple]
    error: str = ""
    tables_used: List[str] = None
    
    def to_dict(self):
        return {
            "success": self.success,
            "columns": self.columns,
            "rows": [list(row) for row in self.rows],
            "error": self.error,
            "tables_used": self.tables_used or []
        }


class SQLiteTool:
    """Tool for interacting with Northwind SQLite database."""
    
    def __init__(self, db_path: str = "data/northwind.sqlite"):
        self.db_path = db_path
        self.conn = None
        self.schema_cache = None
        self._connect()
    
    def _connect(self):
        """Start database connection"""
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row
            
            # Create orderitems view
            cursor = self.conn.cursor()
            cursor.execute('CREATE VIEW IF NOT EXISTS orderitems AS SELECT * FROM "Order Details";')
            self.conn.commit()
            
            print(f"Connected to database: {self.db_path}")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to database: {e}")
    
    def get_schema(self, force_refresh: bool = False) -> Dict[str, List[Tuple[str, str]]]:
        """Get database schema (table names and columns)."""
        if self.schema_cache and not force_refresh:
            return self.schema_cache
        
        cursor = self.conn.cursor()
        
        # Valid tables/views to expose
        valid_tables = ['categories', 'orderitems', 'orders', 'products', 'customers']
        
        # Get all tables
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type IN ('table','view') AND name NOT LIKE 'sqlite_%'
            ORDER BY name
        """)

        sch = {
  "Categories": ["CategoryID", "CategoryName"],
  "Products": ["ProductID", "ProductName", "CategoryID", "UnitPrice"],
  "order_items": ["OrderID", "ProductID", "UnitPrice", "Quantity", "Discount"],
  "Orders": ["OrderID", "CustomerID", "OrderDate"],
  "Customers": ["CustomerID", "CompanyName"]
}
        imp_cols = []
        for _,cols in sch.items():
            for col in cols:
                imp_cols.append(col)

        tables = [row[0] for row in cursor.fetchall() if row[0].lower() in valid_tables]
        
        # Unwanted columns
        unwanted = ['supplierid', 'shipperid', 'picture', 'employeeid',
                   'freight', 'shipname', 'shippostalcode', 'shipaddress',
                   'shipcity', 'shipregion', 'shipcountry']
        
        # Get columns for each table
        schema = {}
        for table in tables:
            cursor.execute(f"PRAGMA table_info({table})")
            columns = [row[1] for row in cursor.fetchall() 
                      if row[1] in imp_cols]
            schema[table] = columns
        
        self.schema_cache = schema
        return schema
    
    def format_schema_for_llm(self) -> str:
        """Format schema """
        schema = self.get_schema()
        list_schema = ["Database Schema (Northwind SQLite):"]
        for table , columns in schema.items():
            
            row = f"Table: {table} includes the following columns: ({', '.join(columns)})"
            list_schema.append(row)

                
        return "\n".join(list_schema)
    
    def extract_tables_from_sql(self, sql: str) -> List[str]:
        """Extract table names from SQL query."""
        sql_upper = sql.upper()
        schema = self.get_schema()
        tables_used = []
        
        for table in schema.keys():
            patterns = [
                f' {table.upper()} ',
                f' {table.upper()},',
                f'FROM {table.upper()}',
                f'JOIN {table.upper()}',
                f'"{table}"',
                f'`{table}`'
            ]
            
            if any(pattern in sql_upper or pattern in sql for pattern in patterns):
                tables_used.append(table)
        
        return list(set(tables_used))
    
    def execute_query(self, sql: str) -> SQLResult:
        """Execute SQL query and return structured result."""
        cursor = self.conn.cursor()
        
        try:
            cursor.execute(sql)
            rows = cursor.fetchall()
            
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            rows_tuples = [tuple(row) for row in rows]
            tables_used = self.extract_tables_from_sql(sql)
            
            return SQLResult(
                success=True,
                columns=columns,
                rows=rows_tuples,
                tables_used=tables_used
            )
        
        except sqlite3.Error as e:
            return SQLResult(
                success=False,
                columns=[],
                rows=[],
                error=str(e)
            )
        
        except Exception as e:
            return SQLResult(
                success=False,
                columns=[],
                rows=[],
                error=f"Unexpected error: {str(e)}"
            )
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            print("Database connection closed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()