-- New tables for memory-aware realtime conversation.

CREATE TABLE user_memory (
    memory_id BIGSERIAL PRIMARY KEY,
    user_id BIGINT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    memory_type VARCHAR(30) NOT NULL CHECK (memory_type IN (
        'RECENT_SUMMARY', 'SPECIAL_EVENT', 'PREFERENCE', 'PERSON', 'PLACE',
        'GOAL', 'FACT', 'REMINDER', 'OTHER'
    )),
    title VARCHAR(120),
    content TEXT NOT NULL,
    event_date DATE,
    importance SMALLINT NOT NULL DEFAULT 3 CHECK (importance BETWEEN 1 AND 5),
    source VARCHAR(10) NOT NULL CHECK (source IN ('USER', 'AI', 'SYSTEM')),
    metadata JSONB,
    last_mentioned_at TIMESTAMP,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    deleted_at TIMESTAMP
);

CREATE INDEX idx_user_memory_user_id ON user_memory(user_id, created_at);
CREATE INDEX idx_user_memory_type_event ON user_memory(user_id, memory_type, event_date);

CREATE TABLE user_memory_evidence (
    evidence_id BIGSERIAL PRIMARY KEY,
    memory_id BIGINT NOT NULL REFERENCES user_memory(memory_id) ON DELETE CASCADE,
    source_type VARCHAR(10) NOT NULL CHECK (source_type IN ('DIARY', 'CHAT', 'SYSTEM')),
    source_id BIGINT,
    quote TEXT NOT NULL,
    occurred_at TIMESTAMP,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_user_memory_evidence_memory_id ON user_memory_evidence(memory_id);
