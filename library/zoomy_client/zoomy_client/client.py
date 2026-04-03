import time

from zoomy_client._http import get, post, delete, download, stream_sse
from zoomy_client.types import ZoomyCase, JobStatus, HealthInfo


class ZoomyClient:
    def __init__(self, base_url="http://localhost:8080", token=None):
        self.base_url = base_url.rstrip("/")
        self._headers = {}
        if token:
            self._headers["Authorization"] = f"Bearer {token}"

    def _url(self, path):
        return f"{self.base_url}/api/v1{path}"

    def health(self) -> HealthInfo:
        return get(self._url("/health"), headers=self._headers, timeout=5)

    def list_models(self) -> list[dict]:
        return get(self._url("/models"), headers=self._headers)

    def submit(self, case: ZoomyCase) -> str:
        resp = post(self._url("/jobs"), data=case, headers=self._headers)
        return resp["job_id"]

    def status(self, job_id: str) -> JobStatus:
        return get(self._url(f"/jobs/{job_id}"), headers=self._headers)

    def list_jobs(self) -> list[JobStatus]:
        return get(self._url("/jobs"), headers=self._headers)

    def results(self, job_id: str) -> dict:
        return get(self._url(f"/jobs/{job_id}/results/fields"), headers=self._headers)

    def download_hdf5(self, job_id: str, output_path: str):
        download(self._url(f"/jobs/{job_id}/results/hdf5"), output_path, headers=self._headers)

    def stream_logs(self, job_id: str):
        yield from stream_sse(self._url(f"/jobs/{job_id}/logs"), headers=self._headers)

    def cancel(self, job_id: str):
        return delete(self._url(f"/jobs/{job_id}"), headers=self._headers)

    def wait(self, job_id: str, interval=2.0, timeout=600) -> JobStatus:
        start = time.time()
        while time.time() - start < timeout:
            s = self.status(job_id)
            if s["status"] in ("complete", "failed"):
                return s
            time.sleep(interval)
        raise TimeoutError(f"Job {job_id} did not complete within {timeout}s")
