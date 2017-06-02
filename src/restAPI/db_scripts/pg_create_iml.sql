--ONLY EVER USE THESE 2 LINES ON DEVELOPMENT MACHINES !!!!!
DROP TABLE IF EXISTS iml_requests_raw;
DROP TABLE IF EXISTS iml_results;


CREATE TABLE IF NOT EXISTS iml_requests_raw (
    id SERIAL PRIMARY KEY,
    request_raw JSON NOT NULL
);


CREATE TABLE IF NOT EXISTS iml_results (
    id SERIAL PRIMARY KEY,
    timestamp BIGINT NOT NULL,
    grouptoken VARCHAR(255) NOT NULL,
    usertoken VARCHAR(255) NOT NULL,
    weights_bias VARCHAR(1024) NOT NULL,
    weights_iml VARCHAR(1024) NOT NULL,
    target VARCHAR(255) NOT NULL,
    user_info JSON,
    survey JSON,
    overall_result JSON NOT NULL
);