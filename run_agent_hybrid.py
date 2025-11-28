import click
import json
from pathlib import Path
from rich.console import Console
from rich.progress import track
import dspy

from agent.dspy_signatures import (
    RouterModule, SQLGeneratorModule, SQLRepairModule, SynthesizerModule 
)
from agent.rag.retreive import BM25Retriever
from agent.tools.sqlite_tool import SQLiteTool
from agent.graph_hybrid import HybridAgent

console = Console()


def setup_dspy():
    """Setup DSPy with Ollama backend"""
    lm_phi = dspy.LM(
        model='ollama/phi3.5:3.8b-mini-instruct-q4_K_M',
        base_url='http://localhost:11434',
        max_tokens= 1500
    )
    dspy.configure(lm=lm_phi)

# Load optimized bootstraped router 
def load_optimized_modules():
    """Load optimized modules if available"""
    optimized_dir = Path("optimized_modules")
    
    if not optimized_dir.exists():
        console.print("[yellow]No optimized modules found, using base modules[/yellow]")
        return None, None
    
    try:
        router = RouterModule()
        sql_gen = SQLGeneratorModule()
        
        router_path = optimized_dir / "router.json"
        
        if router_path.exists():
            router.load(str(router_path))
            console.print("[green]✓ Loaded optimized Router[/green]")
        
        return router, sql_gen
    
    except Exception as e:
        console.print(f"[yellow]Failed to load optimized modules: {e}[/yellow]")
        return None, None


@click.command()
@click.option('--batch', required=True, help='Path to questions JSONL file')
@click.option('--out', required=True, help='Path to output JSONL file')
def main(batch: str, out: str):
    console.print("[bold cyan]🤖 Retail Analytics Copilot[/bold cyan]\n")
    
    # Setup DSPy
    console.print("[yellow]Setting up DSPy with Ollama...[/yellow]")
    setup_dspy()
    
    # Load components
    console.print("[yellow]Loading retriever and database...[/yellow]")
    retriever = BM25Retriever(docs_dir="docs")
    db_tool = SQLiteTool(db_path="data/northwind.sqlite")
    
    # Load questions
    with open(batch, 'r') as f:
        questions = [json.loads(line) for line in f]
    
    console.print(f"[green]Loaded {len(questions)} questions[/green]\n")
    
    # Try to load optimized modules
    console.print("[yellow]Loading modules...[/yellow]")
    opt_router, opt_sql_gen = load_optimized_modules()
    
    # Use optimized if available, otherwise base modules
    router = opt_router if opt_router else RouterModule()
    sql_gen = SQLGeneratorModule()
    sql_repair = SQLRepairModule()
    synth = SynthesizerModule()
    
    # Create agent
    agent = HybridAgent(
        retriever=retriever,
        db_tool=db_tool,
        router=router,
        sql_gen=sql_gen,
        sql_repair=sql_repair,
        synth=synth,
    )
    
    # Process questions
    console.print("\n[bold]Processing questions...[/bold]\n")
    results = []
    
    for q in track(questions, description="Processing"):
        console.print(f"\n[cyan]Q: {q['question'][:80]}...[/cyan]")
        
        try:
            answer = agent.run(
                question=q['question'],
                question_id=q['id'],
                format_hint=q.get('format_hint', 'str')
            )
            
            output = {
                "id": q["id"],
                "final_answer": answer["final_answer"],
                "sql": answer["sql"],
                "confidence": answer["confidence"],
                "explanation": answer["explanation"],
                "citations": answer["citations"]
            }
            
            results.append(output)
            
            console.print(f"[green]A: {answer['final_answer']}[/green]")
            console.print(f"[dim]Confidence: {answer['confidence']:.2f}[/dim]")
        
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            import traceback
            traceback.print_exc()
            
            results.append({
                "id": q["id"],
                "final_answer": None,
                "sql": "",
                "confidence": 0.0,
                "explanation": f"Error: {str(e)}",
                "citations": []
            })
    
    # Save results
    console.print(f"\n[yellow]Saving results to {out}...[/yellow]")
    with open(out, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    
    console.print(f"[bold green]✓ Complete! Results saved to {out}[/bold green]")
    
    # Close database
    db_tool.close()


if __name__ == '__main__':
    main()