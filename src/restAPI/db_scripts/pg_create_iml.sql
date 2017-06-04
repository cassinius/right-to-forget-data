--ONLY EVER USE THESE 2 LINES ON DEVELOPMENT MACHINES !!!!!
DROP TABLE IF EXISTS iml_requests_raw;
DROP TABLE IF EXISTS iml_results;


CREATE TABLE IF NOT EXISTS iml_requests_raw (
    id SERIAL PRIMARY KEY,
    timestamp BIGINT NOT NULL,
    request_raw JSON NOT NULL
);


CREATE TABLE IF NOT EXISTS iml_results (
    id SERIAL PRIMARY KEY,
    timestamp BIGINT NOT NULL,
    grouptoken VARCHAR(255) NOT NULL,
    usertoken VARCHAR(255) NOT NULL,
    target VARCHAR(255) NOT NULL,
    weights_bias VARCHAR(1024) NOT NULL,
    weights_iml VARCHAR(1024) NOT NULL,
    results_bias JSON NOT NULL,
    results_iml JSON NOT NULL,
    plot_url VARCHAR(255) NOT NULL,
    user_info JSON,
    survey JSON
);


ALTER TABLE iml_requests_raw OWNER TO iml_admin;
ALTER TABLE iml_results OWNER TO iml_admin;


-- Sample
--INSERT INTO iml_requests_raw (timestamp, request_raw) VALUES (01341503451, '{ "1": 2, "arr": [1, 2, 3], "bla": "hoo" }');
