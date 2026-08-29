DROP INDEX IF EXISTS ix_chunks_search_vector;

ALTER TABLE document_chunks
    DROP COLUMN search_vector;

ALTER TABLE document_chunks
    ADD COLUMN search_vector tsvector
    GENERATED ALWAYS AS (
        setweight(
            to_tsvector('english', coalesce(section_title, '')),
            'A'
        )
        || setweight(
            to_tsvector('english', coalesce(passage_text, '')),
            'B'
        )
    ) STORED;

CREATE INDEX ix_chunks_search_vector
    ON document_chunks USING gin (search_vector);
