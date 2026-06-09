from fastapi.testclient import TestClient

from content.store import clear_generated_posts


def test_generate_list_and_update_linkedin_draft(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fictional-test-key")
    clear_generated_posts()

    from api.approval_api import app

    client = TestClient(app)
    brief = {
        "topic": "Reducing prior authorization friction",
        "audience": "healthcare operations leaders",
        "content_pillar": "operational efficiency",
        "post_type": "thought leadership post",
        "tone": "clear and practical",
        "cta": "Talk with your team about where authorization work gets stuck.",
        "visual_type": "carousel",
    }

    create_response = client.post("/content/linkedin/drafts/generate", json=brief)

    assert create_response.status_code == 200
    created = create_response.json()
    assert created["id"]
    assert "Reducing prior authorization friction" in created["caption"]
    assert created["visual_title"]
    assert created["visual_sections"]
    assert created["alt_text"]
    assert created["hashtags"]
    assert created["approval_status"] in {"draft", "needs_review"}

    list_response = client.get("/content/linkedin/drafts")
    assert list_response.status_code == 200
    assert [draft["id"] for draft in list_response.json()] == [created["id"]]

    update_response = client.patch(
        f"/content/linkedin/drafts/{created['id']}/approval-status",
        json={"approval_status": "approved"},
    )
    assert update_response.status_code == 200
    assert update_response.json()["approval_status"] == "approved"


def test_invalid_linkedin_draft_approval_status_returns_clear_error(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fictional-test-key")
    clear_generated_posts()

    from api.approval_api import app

    client = TestClient(app)
    response = client.patch(
        "/content/linkedin/drafts/fictional-draft/approval-status",
        json={"approval_status": "auto_post_now"},
    )

    assert response.status_code == 422
    assert "approval_status must be one of" in response.text
