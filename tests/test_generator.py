from src.generation.generator import GroundedGenerator


def test_generator_with_context():
    mock_chunks = [
        {
            "id": "1",
            "content": "API tokens expire after 15 minutes of inactivity.",
            "metadata": {"source": "auth.md"}
        }
    ]
    generator = GroundedGenerator()
    result = generator.generate("How long do tokens last?", mock_chunks)

    assert result.answer is not None
    assert len(result.answer) > 0
    # Must include citation bracket [1] and mention 15 minutes
    assert "[1]" in result.answer
    assert "15" in result.answer


def test_generator_insufficient_context():
    mock_chunks = [
        {
            "id": "1",
            "content": "Our headquarters is located in San Francisco.",
            "metadata": {"source": "about.md"}
        }
    ]
    generator = GroundedGenerator()
    result = generator.generate("What is the payment processing fee?", mock_chunks)

    # Should trigger fallback behavior rather than inventing fees
    assert "INSUFFICIENT_CONTEXT" in result.answer or "does not contain" in result.answer.lower()