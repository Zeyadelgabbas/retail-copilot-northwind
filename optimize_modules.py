import json
import dspy
from pathlib import Path
from rich.console import Console
from rich.table import Table

from agent.dspy_signatures import RouterModule

console = Console()


def setup_dspy():
    """Setup DSPy with Ollama backend"""
    lm = dspy.LM(
        model='ollama/phi3.5:3.8b-mini-instruct-q4_K_M',
        base_url='http://localhost:11434',
        max_tokens=500
    )
    dspy.configure(lm=lm)


def load_training_data(file_path: str):
    """Load training examples from JSONL - handle nested format"""
    examples = []
    
    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            
            # Handle nested format: {"id": 0, "response": {"final_output": [...]}}
            if 'response' in data and 'final_output' in data['response']:
                for ex in data['response']['final_output']:
                    if 'question' in ex and 'route' in ex:
                        examples.append({
                            'question': ex['question'],
                            'route': ex['route'].lower().strip()
                        })
            
            # Handle flat format: {"question": "...", "route": "..."}
            elif 'question' in data and 'route' in data:
                examples.append({
                    'question': data['question'],
                    'route': data['route'].lower().strip()
                })
    
    console.print(f"[green]Loaded {len(examples)} training examples[/green]")
    return examples


def create_dspy_examples(training_data):
    """Convert to DSPy format"""
    dspy_examples = []
    
    for ex in training_data:
        route = ex['route'].lower().strip()
        
        # Validate route
        if route not in ['rag', 'sql', 'hybrid']:
            console.print(f"[yellow]Skipping invalid route: {route} for: {ex['question'][:50]}[/yellow]")
            continue
        
        dspy_examples.append(
            dspy.Example(
                question=ex['question'],
                route=route
            ).with_inputs('question')
        )
    
    return dspy_examples


def evaluate_router(router, test_examples):
    """Evaluate router accuracy"""
    correct = 0
    total = len(test_examples)
    
    console.print(f"\n[cyan]Evaluating on {total} examples...[/cyan]\n")
    
    for ex in test_examples:
        pred = router(question=ex.question)
        expected = ex.route.lower()
        predicted = pred.route.lower()
        
        if predicted == expected:
            correct += 1
            console.print(f"  ✓ {ex.question[:60]}... -> {predicted}")
        else:
            console.print(f"  ✗ {ex.question[:60]}... -> {predicted} (expected {expected})")
    
    accuracy = correct / total if total > 0 else 0
    return accuracy


def main():
    console.print("[bold cyan]Router Optimization with BootstrapFewShot[/bold cyan]\n")
    
    # Setup
    setup_dspy()
    
    # Load training data
    train_path = "data/training_data.jsonl"
    if not Path(train_path).exists():
        console.print(f"[red]Training data not found at {train_path}[/red]")
        return
    
    training_data = load_training_data(train_path)
    
    if len(training_data) < 5:
        console.print(f"[red]Need at least 5 training examples, got {len(training_data)}[/red]")
        return
    
    # Convert to DSPy format
    dspy_examples = create_dspy_examples(training_data)
    console.print(f"[green]Created {len(dspy_examples)} DSPy examples[/green]\n")
    
    if len(dspy_examples) < 5:
        console.print(f"[red]After validation, only {len(dspy_examples)} examples remain (need ≥5)[/red]")
        return
    
    # Split train/test (80/20)
    split_idx = int(len(dspy_examples) * 0.8)
    train_set = dspy_examples[:split_idx]
    test_set = dspy_examples[split_idx:]
    
    console.print(f"[cyan]Train: {len(train_set)}, Test: {len(test_set)}[/cyan]\n")
    
    # Evaluate BASE router
    console.print("[bold yellow]Evaluating BASE Router...[/bold yellow]")
    base_router = RouterModule()
    base_accuracy = evaluate_router(base_router, test_set)
    
    console.print(f"\n[green]Base Accuracy: {base_accuracy:.1%}[/green]\n")
    
    # Optimize with BootstrapFewShot
    console.print("[bold yellow]Optimizing Router with BootstrapFewShot...[/bold yellow]")
    
    def route_metric(example, pred, trace=None):
        """Simple exact match metric"""
        return 1.0 if pred.route.lower() == example.route.lower() else 0.0
    
    try:
        from dspy.teleprompt import BootstrapFewShot
        
        optimizer = BootstrapFewShot(
            metric=route_metric,
            max_bootstrapped_demos=5,  
            max_labeled_demos=3,
            max_rounds=10
        )
        
        console.print("[yellow]Running optimization (may take 2-3 minutes)...[/yellow]")
        optimized_router = optimizer.compile(base_router, trainset=train_set)
        
        # Evaluate OPTIMIZED router
        console.print("\n[bold yellow]Evaluating OPTIMIZED Router...[/bold yellow]")
        opt_accuracy = evaluate_router(optimized_router, test_set)
        
        console.print(f"\n[green]Optimized Accuracy: {opt_accuracy:.1%}[/green]\n")
        
        # Show comparison
        table = Table(title="Optimization Results", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Base", justify="right", style="yellow")
        table.add_column("Optimized", justify="right", style="green")
        table.add_column("Improvement", justify="right", style="blue")
        
        improvement = (opt_accuracy - base_accuracy) * 100
        table.add_row(
            "Accuracy",
            f"{base_accuracy:.1%}",
            f"{opt_accuracy:.1%}",
            f"{improvement:+.1f}%"
        )
        
        console.print(table)
        
        # Save optimized router
        output_dir = Path("optimized_modules")
        output_dir.mkdir(exist_ok=True)
        
        optimized_router.save(str(output_dir / "router.json"))
        
        metrics = {
            "base_accuracy": base_accuracy,
            "optimized_accuracy": opt_accuracy,
            "improvement": improvement,
            "training_examples": len(train_set),
            "test_examples": len(test_set)
        }
        
        with open(output_dir / "metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)
        
        console.print(f"\n[bold green]✓ Saved to {output_dir}/[/bold green]")
    
    except Exception as e:
        console.print(f"[red]Optimization failed: {e}[/red]")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()