from fastapi.testclient import TestClient

from submissions.adapters import EmailSubmissionAdapter, EmailSubmissionConfig
from submissions import registry


def test_legacy_prior_auth_form_email_endpoint_preserves_response(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fictional-test-key")
    monkeypatch.delenv("SMTP_HOST", raising=False)
    monkeypatch.delenv("PRIOR_AUTH_TEST_EMAIL", raising=False)
    sent_payloads = []

    def fake_sender(**kwargs):
        sent_payloads.append(kwargs)
        return {
            "delivery_status": "sent",
            "delivery_mode": "mock",
            "to_email": kwargs["to_email"],
            "attachment_filename": kwargs["attachments"][0]["filename"],
            "attachment_filenames": [item["filename"] for item in kwargs["attachments"]],
        }

    monkeypatch.setitem(
        registry._ADAPTERS,
        "email",
        EmailSubmissionAdapter(
            sender=fake_sender,
            config=EmailSubmissionConfig(default_to_email=None, smtp_configured=False),
        ),
    )

    from api.approval_api import app

    client = TestClient(app)
    response = client.post(
        "/prior-auth-forms/send-email",
        json={
            "to_email": "demo-recipient@example.test",
            "form": {
                "case_id": "fictional-case-legacy",
                "patient_name": "Jay Doe",
                "patient_dob": "01/01/1980",
                "member_id": "FICTIONAL123",
                "payer": "humana",
                "provider_name": "Demo Provider",
                "provider_npi": "1234567890",
                "provider_phone": "555-0100",
                "cpt_code": "64640",
                "diagnosis_codes": ["M79.2"],
                "requested_service": "Peripheral nerve injection",
                "clinical_summary": "Fictional clinical summary for regression testing.",
            },
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["delivery_status"] == "sent"
    assert body["delivery_mode"] == "mock"
    assert body["to_email"] == "demo-recipient@example.test"
    assert body["attachment_filename"] == "prior-auth-form-fictional-case-legacy.pdf"
    assert "outbox_path" in body
    assert sent_payloads[0]["attachments"][0]["filename"] == "prior-auth-form-fictional-case-legacy.pdf"
