from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import embeddings


def _embedding_response(*values: list[float]):
    return SimpleNamespace(
        data=[
            SimpleNamespace(index=index, embedding=value)
            for index, value in reversed(list(enumerate(values)))
        ]
    )


class EmbeddingRequestTests(unittest.TestCase):
    @patch("embeddings.client")
    def test_get_embeddings_applies_optional_request_bounds(self, client):
        bounded_client = Mock()
        bounded_client.embeddings.create.return_value = _embedding_response(
            [1.0, 2.0],
            [3.0, 4.0],
        )
        client.with_options.return_value = bounded_client

        result = embeddings.get_embeddings(
            ["first", "second"],
            timeout_seconds=30,
            max_retries=0,
        )

        self.assertEqual(result, [[1.0, 2.0], [3.0, 4.0]])
        client.with_options.assert_called_once_with(timeout=30, max_retries=0)
        bounded_client.embeddings.create.assert_called_once_with(
            model=embeddings.EMBEDDING_MODEL,
            input=["first", "second"],
        )

    @patch("embeddings.client")
    def test_get_embeddings_preserves_existing_client_behavior(self, client):
        client.embeddings.create.return_value = _embedding_response([1.0])

        self.assertEqual(embeddings.get_embeddings(["query"]), [[1.0]])
        client.with_options.assert_not_called()

    def test_get_embeddings_rejects_invalid_request_bounds(self):
        invalid_values = (
            ("timeout_seconds", 0, "timeout_seconds must be positive"),
            ("max_retries", -1, "max_retries cannot be negative"),
        )
        for keyword, value, message in invalid_values:
            with self.subTest(keyword=keyword):
                with self.assertRaisesRegex(ValueError, message):
                    embeddings.get_embeddings(["query"], **{keyword: value})


if __name__ == "__main__":
    unittest.main()
