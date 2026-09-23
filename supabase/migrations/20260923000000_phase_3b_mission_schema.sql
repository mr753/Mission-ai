-- Phase 3B: Database Schema + Migration for Mission AI
-- Baseline: Locked Phase 1-2E
-- Target: missions, mission_jobs, job_outputs, pipeline_events

-- 0. Helper function for updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- 1. MISSIONS
-- Representasi database dari "MissionContext"
CREATE TABLE IF NOT EXISTS missions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    mission_id TEXT UNIQUE NOT NULL,
    instructions TEXT,
    main_message TEXT NOT NULL,
    key_points JSONB NOT NULL DEFAULT '[]',
    platforms JSONB NOT NULL DEFAULT '[]',
    max_hashtags INTEGER NOT NULL CHECK (max_hashtags >= 0),
    required_hashtags JSONB NOT NULL DEFAULT '[]',
    image_source_url TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL
);

DROP TRIGGER IF EXISTS update_missions_updated_at ON missions;
CREATE TRIGGER update_missions_updated_at
    BEFORE UPDATE ON missions
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

COMMENT ON TABLE missions IS 'Stores mission context and configuration.';
COMMENT ON COLUMN missions.mission_id IS 'Business identifier for the mission.';

-- 2. MISSION_JOBS
-- Representasi job individual
CREATE TABLE IF NOT EXISTS mission_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id TEXT UNIQUE NOT NULL,
    mission_id TEXT NOT NULL REFERENCES missions(mission_id) ON DELETE CASCADE,
    source_path TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'PENDING' CHECK (status IN (
        'PENDING', 'PROCESSING', 'COMPLETED', 'FAILED', 'AUTO_PUBLISHED', 'MANUAL_READY'
    )),
    job_order INTEGER NOT NULL,
    processing_started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    last_error TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL
);

DROP TRIGGER IF EXISTS update_mission_jobs_updated_at ON mission_jobs;
CREATE TRIGGER update_mission_jobs_updated_at
    BEFORE UPDATE ON mission_jobs
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE INDEX idx_mission_jobs_mission_id ON mission_jobs(mission_id);
CREATE INDEX idx_mission_jobs_status ON mission_jobs(status);
CREATE INDEX idx_mission_jobs_stale_recovery ON mission_jobs(status, processing_started_at) 
    WHERE (status = 'PROCESSING');

COMMENT ON TABLE mission_jobs IS 'Individual processing jobs for images within a mission.';
COMMENT ON COLUMN mission_jobs.job_id IS 'Deterministic ID: mission_id + _ + SHA-256(image).';

-- 3. JOB_OUTPUTS
-- Satu output untuk job
CREATE TABLE IF NOT EXISTS job_outputs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id TEXT UNIQUE NOT NULL REFERENCES mission_jobs(job_id) ON DELETE CASCADE,
    analysis JSONB,
    captions JSONB NOT NULL DEFAULT '[]',
    video_path TEXT,
    status TEXT NOT NULL CHECK (status IN (
        'PENDING', 'PROCESSING', 'COMPLETED', 'FAILED', 'AUTO_PUBLISHED', 'MANUAL_READY'
    )),
    error TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL
);

DROP TRIGGER IF EXISTS update_job_outputs_updated_at ON job_outputs;
CREATE TRIGGER update_job_outputs_updated_at
    BEFORE UPDATE ON job_outputs
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE INDEX idx_job_outputs_job_id ON job_outputs(job_id);

COMMENT ON TABLE job_outputs IS 'Stores the resulting content package from a successful or failed job.';

-- 4. PIPELINE_EVENTS
-- Event/audit log terstruktur untuk pipeline
-- Retention policy: Audit events harus bertahan meskipun job dihapus.
-- Mission deletion di-RESTRICT jika masih ada history audit untuk mencegah kehilangan data audit secara tidak sengaja.
CREATE TABLE IF NOT EXISTS pipeline_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    mission_id TEXT NOT NULL REFERENCES missions(mission_id) ON DELETE RESTRICT,
    job_id TEXT REFERENCES mission_jobs(job_id) ON DELETE SET NULL,
    event_type TEXT NOT NULL,
    status TEXT, -- Optional as per Phase 3A
    payload JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL
);

CREATE INDEX idx_pipeline_events_mission_id ON pipeline_events(mission_id);
CREATE INDEX idx_pipeline_events_job_id ON pipeline_events(job_id);
CREATE INDEX idx_pipeline_events_created_at ON pipeline_events(created_at);

COMMENT ON TABLE pipeline_events IS 'Structured audit logs for pipeline execution stages.';

-- RLS / SECURITY
-- default-deny policy (enable RLS without creating any policies)

ALTER TABLE missions ENABLE ROW LEVEL SECURITY;
ALTER TABLE mission_jobs ENABLE ROW LEVEL SECURITY;
ALTER TABLE job_outputs ENABLE ROW LEVEL SECURITY;
ALTER TABLE pipeline_events ENABLE ROW LEVEL SECURITY;

-- Note: No public policies are created. Access requires service-role or specific policies.
