ALTER TABLE documents
    ADD COLUMN IF NOT EXISTS chunking_config jsonb NOT NULL DEFAULT '{}'::jsonb;

UPDATE documents
SET chunking_config = jsonb_build_object(
        'max_tokens', 200,
        'overlap_tokens', 30,
        'min_chunk_tokens', 40,
        'encoding_name', 'cl100k_base'
    ),
    updated_at = now()
WHERE chunking_version = 'structure-token-v1'
  AND chunking_config = '{}'::jsonb;

ALTER TABLE documents
    ADD CONSTRAINT ck_documents_chunking_config
    CHECK (jsonb_typeof(chunking_config) = 'object');
