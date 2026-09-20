import io

import pytest
from fastapi.testclient import TestClient

from personalens import HashEmbedder
from personalens.api.app import create_app

from conftest import BAYES


@pytest.fixture
def client(settings):
    return TestClient(create_app(settings, HashEmbedder()))


def test_upload_search_list_delete(client):
    r = client.post("/documents", data={"user_id": "u1"}, files={"file": ("bayes.md", io.BytesIO(BAYES.encode()))})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "ingested" and body["n_chunks"] >= 1

    dup = client.post("/documents", data={"user_id": "u1"}, files={"file": ("bayes.md", io.BytesIO(BAYES.encode()))})
    assert dup.json()["status"] == "duplicate"

    hits = client.get("/memory/search", params={"user_id": "u1", "q": "posterior prior"}).json()
    assert hits and hits[0]["chunk"]["doc_id"] == body["doc_id"]

    assert len(client.get("/documents", params={"user_id": "u1"}).json()) == 1
    assert client.delete(f"/documents/{body['doc_id']}", params={"user_id": "other"}).status_code == 404
    assert client.delete(f"/documents/{body['doc_id']}", params={"user_id": "u1"}).status_code == 200


def test_csv_upload_routes_to_events(client):
    csv = "timestamp,concept,correct\n2026-09-01,Bayes,1\n"
    r = client.post("/documents", data={"user_id": "u1"}, files={"file": ("quiz.csv", io.BytesIO(csv.encode()))})
    assert r.status_code == 200 and r.json()["status"] == "events_ingested" and r.json()["n_events"] == 1


def test_error_codes(client):
    bad_type = client.post("/documents", data={"user_id": "u1"}, files={"file": ("x.exe", io.BytesIO(b"x"))})
    assert bad_type.status_code == 415
    bad_csv = client.post("/documents", data={"user_id": "u1"}, files={"file": ("q.csv", io.BytesIO(b"a,b\n1,2\n"))})
    assert bad_csv.status_code == 422
    bad_src = client.post("/documents", data={"user_id": "u1", "source_type": "quiz"}, files={"file": ("a.md", io.BytesIO(BAYES.encode()))})
    assert bad_src.status_code == 422
