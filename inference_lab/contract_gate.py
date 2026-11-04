"""Evidence-span and gold-field gates for five fictional contract snippets.

This checks extraction mechanics, not legal sufficiency or the truth of a claim.
"""
import argparse
import json
from pathlib import Path


def grounded_json_gate(text, source, expected):
    try:
        parsed = json.loads(text)
    except (ValueError, TypeError, RecursionError):
        return {"pass": False, "reason": "invalid_json"}
    if not isinstance(parsed, dict) or set(parsed) != set(expected):
        return {"pass": False, "reason": "field_schema"}
    grounded, correct, errors = 0, 0, []
    for name, gold in expected.items():
        value = parsed[name]
        if not isinstance(value, dict) or set(value) != {"value", "quote"}:
            errors.append(name + ":schema")
            continue
        field, quote = value['value'], value['quote']
        if field is None and quote is None:
            grounded += 1
        elif isinstance(field, str) and field and isinstance(quote, str) and quote and quote in source and field in quote:
            grounded += 1
        else:
            errors.append(name + ":ungrounded")
        if field == gold:
            correct += 1
        else:
            errors.append(name + ":incorrect")
    return {"pass": grounded == len(expected) and correct == len(expected),
            "grounded_fields": grounded, "correct_fields": correct,
            "fields": len(expected), "errors": errors,
            "scope": "exact spans and fixed gold fields; not semantic entailment or legal advice"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidates", type=Path, help="JSONL objects with case_id and output (JSON string)")
    parser.add_argument("--workload", type=Path,
                        default=Path(__file__).resolve().parents[1]/'data'/'contract_extraction.jsonl')
    args = parser.parse_args()
    workload = [json.loads(line) for line in args.workload.read_text(encoding='utf-8').splitlines() if line]
    candidates = [json.loads(line) for line in args.candidates.read_text(encoding='utf-8').splitlines() if line]
    ids = [case['case_id'] for case in candidates]
    if len(set(ids)) != len(ids) or set(ids) != {case['case_id'] for case in workload}:
        parser.error('Exactly one candidate per known case is required; missing/error responses remain failures')
    by_id = {case['case_id']: case['output'] for case in candidates}
    results = [dict(case_id=case['case_id'], **grounded_json_gate(by_id[case['case_id']], case['source'], case['expected']))
               for case in workload]
    print(json.dumps({'requests':len(workload), 'passed':sum(r['pass'] for r in results),
                      'results':results, 'provider_execution':'not_established_by_this_gate'}, indent=2))


if __name__ == '__main__':
    main()
