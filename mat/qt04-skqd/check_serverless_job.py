from qiskit_ibm_catalog import QiskitServerless

client = QiskitServerless(name='qdc-2025')

job_id = 'cfc16eb6-ca4f-4215-9609-1d7f889259c1'
print(f"Checking serverless job {job_id}...")

job = client.get_job_by_id(job_id)
print(f"Status: {job.status()}")

result = job.result()
print(f"\nResult keys: {list(result.keys())}")
print(f"\nFull result:")
for key, value in result.items():
    print(f"  {key}: {value}")

# Check logs if available
if 'logs' in result:
    print("\n" + "="*60)
    print("JOB LOGS:")
    print("="*60)
    print(result['logs'])
