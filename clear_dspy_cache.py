"""
Clear DSPy cache to force re-optimization
"""
import shutil
from pathlib import Path
from rich.console import Console

console = Console()

def clear_dspy_cache():
    """Clear DSPy cache directory"""
    cache_dirs = [
        Path.home() / ".dspy_cache",
        Path(".dspy_cache"),
        Path("dspy_cache"),
        Path.home() / ".cache" / "dspy"
    ]
    
    cleared = False
    for cache_dir in cache_dirs:
        if cache_dir.exists():
            console.print(f"[yellow]Removing cache: {cache_dir}[/yellow]")
            shutil.rmtree(cache_dir)
            cleared = True
    
    if not cleared:
        console.print("[green]No cache found (already clean)[/green]")
    else:
        console.print("[green]✓ DSPy cache cleared[/green]")

def clear_optimized_modules():
    """Clear previously optimized modules"""
    opt_dir = Path("optimized_modules")
    if opt_dir.exists():
        console.print(f"[yellow]Removing optimized modules: {opt_dir}[/yellow]")
        shutil.rmtree(opt_dir)
        console.print("[green]✓ Optimized modules removed[/green]")
    else:
        console.print("[green]No optimized modules found[/green]")

if __name__ == '__main__':
    console.print("[bold cyan]Clearing DSPy Cache & Optimized Modules[/bold cyan]\n")
    clear_dspy_cache()
    clear_optimized_modules()
    console.print("\n[bold green]Done! Now run optimize_modules.py[/bold green]")