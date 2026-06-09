from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


DEFAULT_MOCK_HEALTH_BASE_URL = "https://api.mock.health/fhir"


class FHIRClientError(Exception):
    pass


class FHIRClient:
    def __init__(self, base_url: Optional[str] = None, bearer_token: Optional[str] = None):
        self.base_url = (base_url or os.getenv("FHIR_BASE_URL") or DEFAULT_MOCK_HEALTH_BASE_URL).rstrip("/")
        self.bearer_token = (
            bearer_token
            or os.getenv("MOCK_HEALTH_API_KEY")
            or os.getenv("FHIR_BEARER_TOKEN")
        )

    def get_resource(self, resource_type: str, resource_id: str) -> Dict[str, Any]:
        return self.get_json(f"{resource_type}/{resource_id}")

    def search(self, resource_type: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self.get_json(resource_type, params=params or {})

    def search_entries(self, resource_type: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        bundle = self.search(resource_type, params=params or {})
        return [
            entry.get("resource", {})
            for entry in bundle.get("entry", [])
            if isinstance(entry.get("resource"), dict)
        ]

    def get_json(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        query = f"?{urlencode(params)}" if params else ""
        url = f"{self.base_url}/{path.lstrip('/')}{query}"
        headers = {"Accept": "application/fhir+json"}
        if self.bearer_token:
            headers["Authorization"] = f"Bearer {self.bearer_token}"

        request = Request(url, headers=headers)
        try:
            with urlopen(request, timeout=30) as response:
                raw = response.read().decode("utf-8")
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise FHIRClientError(f"FHIR request failed ({exc.code}) for {url}: {detail}") from exc
        except URLError as exc:
            raise FHIRClientError(f"FHIR request failed for {url}: {exc}") from exc

        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise FHIRClientError(f"FHIR response was not JSON for {url}") from exc
