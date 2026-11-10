"""Readable failure notes for GitHub Actions; no-op outside Actions.

Annotations are shown on the run page and in the check-run API, so a failed
daily update can be diagnosed without opening the full step log.
"""
import os
import sys


def _escape(text,*,title=False):
    text=str(text).replace('%','%25').replace('\r','%0D').replace('\n','%0A')
    return text.replace(':','%3A').replace(',','%2C') if title else text


def annotation(title,message):
    return f'::error title={_escape(title,title=True)}::{_escape(message)}'


def report_error(title,message,*,env=None,stream=None):
    """Emit one error annotation and append it to the step summary."""
    env=os.environ if env is None else env
    if env.get('GITHUB_ACTIONS')!='true':return False
    print(annotation(title,message),file=stream or sys.stdout,flush=True)
    summary=env.get('GITHUB_STEP_SUMMARY')
    if summary:
        with open(summary,'a',encoding='utf-8') as f:f.write(f'- **{title}**: `{message}`\n')
    return True
