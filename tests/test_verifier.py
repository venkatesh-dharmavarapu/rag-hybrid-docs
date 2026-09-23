from src.generation.verifier import CitationVerifier


def test_citation_verifier_supported():
    verifier = CitationVerifier()
    source = "All auth tokens expire after 15 minutes of inactivity."
    claim = "Tokens expire after 15 minutes of inactivity."

    is_supported = verifier.verify_citation(claim, source)
    assert is_supported is True


def test_citation_verifier_hallucination():
    verifier = CitationVerifier()
    source = "All auth tokens expire after 15 minutes of inactivity."
    claim = "Tokens cost 5 dollars to regenerate."

    is_supported = verifier.verify_citation(claim, source)
    assert is_supported is False