"""
Allow running postproc modules with python -m postproc.modulename

This allows the postproc scripts to be run as modules while maintaining
their standalone script functionality.
"""
import sys

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m postproc <module> [args...]")
        print("Available modules: echo, noise")
        sys.exit(1)
    
    module_name = sys.argv[1]
    # Remove module name from argv so the submodule sees the right args
    sys.argv = [f"postproc.{module_name}"] + sys.argv[2:]
    
    if module_name == "echo":
        from .echo import main
        main()
    elif module_name == "noise":
        from .noise import main
        main()
    else:
        print(f"Unknown module: {module_name}")
        print("Available modules: echo, noise")
        sys.exit(1)
