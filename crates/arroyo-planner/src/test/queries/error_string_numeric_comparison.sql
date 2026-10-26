--fail=Cannot implicitly combine string and numeric values: inventory_stream.user_id != Int64(123) (Utf8 and Int64). Use matching types or convert explicitly with CAST or TRY_CAST. CAST fails on invalid input; TRY_CAST returns NULL.
CREATE TABLE inventory_stream (user_id TEXT NOT NULL) WITH (
    connector = 'kafka',
    bootstrap_servers = 'localhost:9092',
    topic = 'inventory',
    format = 'json',
    type = 'source'
);

CREATE TABLE inventory_sink WITH (
    connector = 'kafka',
    bootstrap_servers = 'localhost:9092',
    topic = 'inventory-output',
    format = 'json',
    type = 'sink'
);

INSERT INTO inventory_sink
SELECT * FROM inventory_stream WHERE user_id != 123;
