# msiconvert/tools/check_ontology.py
"""Command-line tool to check ontology terms in imzML files."""

import argparse
import logging
from pathlib import Path
import json


from ..metadata.ontology.cache import ONTOLOGY


def main():
    parser = argparse.ArgumentParser(description='Check ontology terms in imzML files')
    parser.add_argument('input', help='imzML file or directory to check')
    parser.add_argument('--output', help='Save results to JSON file')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    
    from ..metadata.validator import ImzMLOntologyValidator
    validator = ImzMLOntologyValidator()
    input_path = Path(args.input)
    
    if input_path.is_file():
        results = validator.validate_file(input_path)
        print(results['summary'])
    else:
        results = validator.validate_directory(input_path)
        print(f"\nChecked {results['files_checked']} files")
        print(f"Found {len(results['all_unknown_terms'])} unique unknown terms")
        
        if results['all_unknown_terms']:
            print("\nMost common unknown terms:")
            for term in list(results['all_unknown_terms'])[:20]:
                print(f"  - {term}")
    
    # Save results if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to {args.output}")
    
    # Show global unknown terms
    print("\n" + ONTOLOGY.report_unknown_terms())


if __name__ == '__main__':
    main()