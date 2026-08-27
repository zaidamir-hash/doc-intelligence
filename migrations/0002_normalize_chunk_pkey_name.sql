DO $$
BEGIN
    IF to_regclass('public.document_chunks_pkey1') IS NOT NULL
       AND to_regclass('public.document_chunks_pkey') IS NULL THEN
        ALTER INDEX document_chunks_pkey1 RENAME TO document_chunks_pkey;
    END IF;
END
$$;
