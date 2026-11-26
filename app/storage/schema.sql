-- Schema definition for code-context-engine SQLite metadata store
-- Schema Version: 1

CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS files (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    path TEXT UNIQUE NOT NULL,
    module_name TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    line_count INTEGER NOT NULL,
    last_modified REAL NOT NULL,
    module_docstring TEXT,
    parse_error TEXT,
    indexed_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS imports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id INTEGER NOT NULL,
    module TEXT NOT NULL,
    name TEXT,
    alias TEXT,
    is_from_import INTEGER NOT NULL,
    level INTEGER NOT NULL,
    lineno INTEGER NOT NULL,
    FOREIGN KEY(file_id) REFERENCES files(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS classes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id INTEGER NOT NULL,
    name TEXT NOT NULL,
    qualified_name TEXT UNIQUE NOT NULL,
    start_line INTEGER NOT NULL,
    end_line INTEGER NOT NULL,
    bases_json TEXT NOT NULL,
    decorators_json TEXT NOT NULL,
    docstring TEXT,
    source TEXT NOT NULL,
    FOREIGN KEY(file_id) REFERENCES files(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS functions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id INTEGER NOT NULL,
    class_id INTEGER,
    name TEXT NOT NULL,
    qualified_name TEXT UNIQUE NOT NULL,
    start_line INTEGER NOT NULL,
    end_line INTEGER NOT NULL,
    arguments_json TEXT NOT NULL,
    return_annotation TEXT,
    decorators_json TEXT NOT NULL,
    docstring TEXT,
    source TEXT NOT NULL,
    is_async INTEGER NOT NULL,
    is_method INTEGER NOT NULL,
    parent_class TEXT,
    FOREIGN KEY(file_id) REFERENCES files(id) ON DELETE CASCADE,
    FOREIGN KEY(class_id) REFERENCES classes(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    function_id INTEGER NOT NULL,
    name TEXT NOT NULL,
    lineno INTEGER NOT NULL,
    FOREIGN KEY(function_id) REFERENCES functions(id) ON DELETE CASCADE
);

-- Required Performance Indexes
CREATE INDEX IF NOT EXISTS idx_files_path ON files(path);
CREATE INDEX IF NOT EXISTS idx_files_content_hash ON files(content_hash);
CREATE INDEX IF NOT EXISTS idx_functions_name ON functions(name);
CREATE INDEX IF NOT EXISTS idx_functions_qualified_name ON functions(qualified_name);
CREATE INDEX IF NOT EXISTS idx_classes_name ON classes(name);
CREATE INDEX IF NOT EXISTS idx_calls_name ON calls(name);
CREATE INDEX IF NOT EXISTS idx_imports_module ON imports(module);
