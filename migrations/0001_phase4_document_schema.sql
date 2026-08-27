CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

DO $$
BEGIN
    IF to_regclass('public.document_chunks') IS NOT NULL
       AND NOT EXISTS (
           SELECT 1
           FROM information_schema.columns
           WHERE table_schema = 'public'
             AND table_name = 'document_chunks'
             AND column_name = 'document_id'
       ) THEN
        ALTER TABLE document_chunks RENAME TO document_chunks_legacy;
    END IF;
END
$$;

CREATE TABLE IF NOT EXISTS documents (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    original_filename varchar NOT NULL,
    content_hash char(64) NOT NULL,
    page_count integer NOT NULL DEFAULT 0,
    chunk_count integer NOT NULL DEFAULT 0,
    status varchar(20) NOT NULL,
    extraction_version varchar NOT NULL,
    chunking_version varchar NOT NULL,
    embedding_model varchar NOT NULL,
    processing_error text,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT uq_documents_filename_hash UNIQUE (original_filename, content_hash),
    CONSTRAINT ck_documents_page_count CHECK (page_count >= 0),
    CONSTRAINT ck_documents_chunk_count CHECK (chunk_count >= 0),
    CONSTRAINT ck_documents_status CHECK (
        status IN ('processing', 'ready', 'failed', 'superseded')
    )
);

CREATE INDEX IF NOT EXISTS ix_documents_status ON documents (status);
CREATE UNIQUE INDEX IF NOT EXISTS uq_documents_ready_filename
    ON documents (original_filename)
    WHERE status = 'ready';

CREATE TABLE IF NOT EXISTS document_chunks (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id uuid NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index integer NOT NULL,
    content text NOT NULL,
    passage_text text NOT NULL,
    page_start integer,
    page_end integer,
    section_title text,
    content_hash char(64) NOT NULL,
    token_count integer NOT NULL,
    overlap_token_count integer NOT NULL DEFAULT 0,
    chunking_version varchar NOT NULL,
    embedding_model varchar NOT NULL,
    boundary_kinds jsonb NOT NULL DEFAULT '[]'::jsonb,
    search_vector tsvector GENERATED ALWAYS AS (
        to_tsvector('english', coalesce(passage_text, ''))
    ) STORED,
    embedding vector(1536) NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT uq_chunks_document_index UNIQUE (document_id, chunk_index),
    CONSTRAINT uq_chunks_document_hash UNIQUE (document_id, content_hash),
    CONSTRAINT ck_chunks_index CHECK (chunk_index >= 0),
    CONSTRAINT ck_chunks_page_start CHECK (page_start IS NULL OR page_start > 0),
    CONSTRAINT ck_chunks_page_end CHECK (
        page_end IS NULL OR page_end >= page_start
    ),
    CONSTRAINT ck_chunks_token_count CHECK (token_count >= 0),
    CONSTRAINT ck_chunks_overlap_count CHECK (overlap_token_count >= 0)
);

CREATE INDEX IF NOT EXISTS ix_chunks_document_id
    ON document_chunks (document_id);
CREATE INDEX IF NOT EXISTS ix_chunks_document_pages
    ON document_chunks (document_id, page_start, page_end);
CREATE INDEX IF NOT EXISTS ix_chunks_search_vector
    ON document_chunks USING gin (search_vector);

DO $$
DECLARE
    expected_chunks bigint;
    migrated_chunks bigint;
BEGIN
    IF to_regclass('public.document_chunks_legacy') IS NULL THEN
        RETURN;
    END IF;

    WITH ranked AS (
        SELECT
            legacy.*,
            row_number() OVER (
                PARTITION BY filename, chunk_index ORDER BY id DESC
            ) AS duplicate_rank
        FROM document_chunks_legacy AS legacy
    ),
    retained AS (
        SELECT * FROM ranked WHERE duplicate_rank = 1
    )
    INSERT INTO documents (
        original_filename,
        content_hash,
        page_count,
        chunk_count,
        status,
        extraction_version,
        chunking_version,
        embedding_model
    )
    SELECT
        filename,
        encode(
            digest(
                string_agg(content, chr(30) ORDER BY chunk_index, id),
                'sha256'
            ),
            'hex'
        ),
        0,
        count(*),
        'ready',
        'legacy-unversioned',
        'legacy-unversioned',
        'text-embedding-3-small'
    FROM retained
    GROUP BY filename;

    WITH ranked AS (
        SELECT
            legacy.*,
            row_number() OVER (
                PARTITION BY filename, chunk_index ORDER BY id DESC
            ) AS duplicate_rank
        FROM document_chunks_legacy AS legacy
    ),
    retained AS (
        SELECT * FROM ranked WHERE duplicate_rank = 1
    )
    INSERT INTO document_chunks (
        document_id,
        chunk_index,
        content,
        passage_text,
        page_start,
        page_end,
        section_title,
        content_hash,
        token_count,
        overlap_token_count,
        chunking_version,
        embedding_model,
        boundary_kinds,
        embedding
    )
    SELECT
        document.id,
        retained.chunk_index,
        retained.content,
        retained.content,
        CASE
            WHEN retained.content ~ '^\[Source pages [0-9]+-[0-9]+\]'
                THEN substring(retained.content FROM '^\[Source pages ([0-9]+)-')::integer
            WHEN retained.content ~ '^\[Source page [0-9]+\]'
                THEN substring(retained.content FROM '^\[Source page ([0-9]+)\]')::integer
            ELSE NULL
        END,
        CASE
            WHEN retained.content ~ '^\[Source pages [0-9]+-[0-9]+\]'
                THEN substring(retained.content FROM '^\[Source pages [0-9]+-([0-9]+)\]')::integer
            WHEN retained.content ~ '^\[Source page [0-9]+\]'
                THEN substring(retained.content FROM '^\[Source page ([0-9]+)\]')::integer
            ELSE NULL
        END,
        substring(retained.content FROM '\[Section: ([^]]+)\]'),
        encode(
            digest(
                retained.filename || chr(30) || retained.chunk_index::text
                || chr(30) || retained.content,
                'sha256'
            ),
            'hex'
        ),
        0,
        0,
        'legacy-unversioned',
        'text-embedding-3-small',
        '["legacy-unversioned"]'::jsonb,
        retained.embedding
    FROM retained
    JOIN documents AS document
      ON document.original_filename = retained.filename
     AND document.chunking_version = 'legacy-unversioned';

    UPDATE documents AS document
    SET page_count = COALESCE(metadata.maximum_page, 0),
        updated_at = now()
    FROM (
        SELECT document_id, max(page_end) AS maximum_page
        FROM document_chunks
        GROUP BY document_id
    ) AS metadata
    WHERE document.id = metadata.document_id;

    SELECT count(*) INTO expected_chunks
    FROM (
        SELECT DISTINCT filename, chunk_index
        FROM document_chunks_legacy
    ) AS distinct_legacy_chunks;

    SELECT count(*) INTO migrated_chunks
    FROM document_chunks
    WHERE chunking_version = 'legacy-unversioned';

    IF migrated_chunks <> expected_chunks THEN
        RAISE EXCEPTION
            'Legacy migration count mismatch: expected %, got %',
            expected_chunks,
            migrated_chunks;
    END IF;

    DROP TABLE document_chunks_legacy;
END
$$;
