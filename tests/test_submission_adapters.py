from submissions import EmailSubmissionAdapter, PriorAuthArtifact, PriorAuthCase, get_submission_adapter


def test_registry_selects_email_adapter():
    adapter = get_submission_adapter("email")

    assert adapter.method == "email"


def test_registry_selects_demo_submission_methods():
    assert get_submission_adapter("pdf_form").method == "pdf_form"
    assert get_submission_adapter("portal_fields").method == "portal_fields"
    assert get_submission_adapter("api_payload").method == "api_payload"


def test_email_adapter_prepares_and_submits_with_mocked_sender(monkeypatch):
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

    adapter = EmailSubmissionAdapter(sender=fake_sender)
    case = PriorAuthCase(
        case_id="fictional-case-001",
        patient_name="Jay Doe",
        payer="humana",
        cpt_code="64640",
    )
    artifact = PriorAuthArtifact(
        filename="completed-pa-form.pdf",
        content=b"%PDF fictional test content",
        mime_type="application/pdf",
    )
    supporting_document = PriorAuthArtifact(
        filename="clinical-note.txt",
        content=b"Fictional clinical note for testing.",
        mime_type="text/plain",
    )

    prepared = adapter.prepare(
        case,
        artifacts=[artifact, supporting_document],
        to_email="demo-recipient@example.test",
    )
    submitted = adapter.submit(prepared)

    assert prepared.method == "email"
    assert "Jay Doe" in prepared.prepared_payload["subject"]
    assert "completed-pa-form.pdf" in prepared.prepared_payload["body"]
    assert submitted.status == "sent"
    assert submitted.metadata["delivery"]["delivery_mode"] == "mock"
    assert sent_payloads == [
        {
            "to_email": "demo-recipient@example.test",
            "subject": prepared.prepared_payload["subject"],
            "body": prepared.prepared_payload["body"],
            "attachments": [
                {
                    "filename": "completed-pa-form.pdf",
                    "content": b"%PDF fictional test content",
                    "mime_type": "application/pdf",
                },
                {
                    "filename": "clinical-note.txt",
                    "content": b"Fictional clinical note for testing.",
                    "mime_type": "text/plain",
                },
            ],
        }
    ]


def test_pdf_portal_and_api_payload_adapters_prepare_outputs():
    case = PriorAuthCase(
        case_id="fictional-case-002",
        patient_name="Jay Doe",
        payer="humana",
        cpt_code="64640",
        provider_name="Demo Provider",
    )
    artifact = PriorAuthArtifact(
        filename="completed-pa-form.pdf",
        content=b"%PDF fictional test content",
        mime_type="application/pdf",
    )

    pdf_result = get_submission_adapter("pdf_form").submit(
        get_submission_adapter("pdf_form").prepare(case, artifacts=[artifact])
    )
    portal_result = get_submission_adapter("portal_fields").submit(
        get_submission_adapter("portal_fields").prepare(case)
    )
    api_result = get_submission_adapter("api_payload").submit(
        get_submission_adapter("api_payload").prepare(case, artifacts=[artifact])
    )

    assert pdf_result.status == "prepared"
    assert pdf_result.metadata["artifact_filenames"] == ["completed-pa-form.pdf"]
    assert portal_result.prepared_payload["fields"]["cptCode"] == "64640"
    assert portal_result.prepared_payload["fields"]["memberName"] == "Jay Doe"
    assert api_result.prepared_payload["payload"]["authorization_request"]["cpt_code"] == "64640"
    assert api_result.prepared_payload["payload"]["artifacts"][0]["filename"] == "completed-pa-form.pdf"
