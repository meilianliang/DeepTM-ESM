import os
import sys
import importlib

sys.path.insert(0, os.path.dirname(__file__))

def main():
    if 'DATASET' in os.environ:
        module_name = 'dataset_DeepSTABp.impl_test'
    else:
        module_name = 'dataset_Tm50.impl_test'

    try:
        impl = importlib.import_module(module_name)
    except ImportError:
        print(f"Error: Cannot find implementation module {module_name}", file=sys.stderr)
        sys.exit(1)

    if hasattr(impl, 'main'):
        sys.exit(impl.main())
    else:
        print(f"Error: Module {module_name} has no main() function", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()