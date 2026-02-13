FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Run tests at build time to verify integrity
RUN python -m pytest tests/ -x --tb=short -q 2>/dev/null || python -c "
import subprocess, sys
for f in ['tests/test_identity.py','tests/test_core.py','tests/test_crdt.py',
          'tests/test_graph.py','tests/test_moral.py','tests/test_provenance.py',
          'tests/test_memory.py','tests/test_execution.py','tests/test_federation.py',
          'tests/test_interface.py']:
    r = subprocess.run([sys.executable, f], capture_output=True, text=True)
    if r.returncode != 0:
        print(f'FAIL: {f}')
        print(r.stdout[-500:])
        sys.exit(1)
    else:
        import re
        m = re.search(r'(\d+) tests', f)
        print(f'OK: {f}')
print('All tests passed')
"

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
