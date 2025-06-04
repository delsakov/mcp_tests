CREATE TABLE code_elements (
    id SERIAL PRIMARY KEY,         -- Unique identifier for each code element
    name VARCHAR(255) NOT NULL,    -- Name of the class, function, etc.
    type VARCHAR(50) NOT NULL,     -- Type of the element (e.g., 'class', 'function', 'method')
    file_path TEXT NOT NULL,       -- Path to the file where the element is defined
    UNIQUE (name, type, file_path) -- Ensure uniqueness for a given element
);

CREATE TABLE dependencies (
    parent_id INTEGER NOT NULL REFERENCES code_elements(id) ON DELETE CASCADE,
    child_id INTEGER NOT NULL REFERENCES code_elements(id) ON DELETE CASCADE,
    PRIMARY KEY (parent_id, child_id) -- Prevents duplicate dependencies
);

-- Optional: Indexes for performance
CREATE INDEX idx_dependencies_parent_id ON dependencies(parent_id);
CREATE INDEX idx_dependencies_child_id ON dependencies(child_id);
CREATE INDEX idx_code_elements_type ON code_elements(type);

--Querying Children Per Level (BFS-like)
WITH RECURSIVE bfs_levels AS (
    -- Anchor member: Select the starting node(s)
    SELECT
        ce.id,
        ce.name,
        ce.type,
        ce.file_path,
        0 AS level  -- Starting level
    FROM
        code_elements ce
    WHERE
        ce.name = 'YourStartFunctionName'  -- Specify the starting node's name
        AND ce.type = 'function'           -- Specify its type
        AND ce.file_path = '/path/to/your/file.py' -- Specify its file path

    UNION ALL

    -- Recursive member: Select children of nodes from the previous level
    SELECT
        child_ce.id,
        child_ce.name,
        child_ce.type,
        child_ce.file_path,
        prev_level.level + 1
    FROM
        code_elements child_ce
    JOIN
        dependencies d ON child_ce.id = d.child_id
    JOIN
        bfs_levels prev_level ON d.parent_id = prev_level.id
    WHERE
        prev_level.level < 10 -- Optional: Limit recursion depth to prevent infinite loops in cyclic graphs (though yours is a DAG)
)
SELECT
    level,
    COUNT(*) AS number_of_children,
    STRING_AGG(name || ' (' || type || ' at ' || file_path || ')', ', ') AS children_details
FROM
    bfs_levels
GROUP BY
    level
ORDER BY
    level;


--To get just the number of children per level without specifying a starting node (i.e., starting from all nodes that are not children themselves - true roots of the DAG), the anchor member of the CTE would need to select all nodes that do not appear as a child_id in the dependencies table.
WITH RECURSIVE bfs_levels AS (
    -- Anchor member: Select all root nodes (nodes that are not children of any other node)
    SELECT
        ce.id,
        ce.name,
        ce.type,
        ce.file_path,
        0 AS level
    FROM
        code_elements ce
    LEFT JOIN
        dependencies d ON ce.id = d.child_id
    WHERE
        d.parent_id IS NULL -- This identifies root nodes

    UNION ALL

    -- Recursive member
    SELECT
        child_ce.id,
        child_ce.name,
        child_ce.type,
        child_ce.file_path,
        prev_level.level + 1
    FROM
        code_elements child_ce
    JOIN
        dependencies d ON child_ce.id = d.child_id
    JOIN
        bfs_levels prev_level ON d.parent_id = prev_level.id
    WHERE prev_level.level < 10 -- Safety limit
)
SELECT
    level,
    COUNT(*) AS number_of_elements_at_level
FROM
    bfs_levels
GROUP BY
    level
ORDER BY
    level;



--SQL Function to Get Descendants with Relative Levels
CREATE OR REPLACE FUNCTION get_node_descendants_with_levels(p_start_node_id INTEGER)
RETURNS TABLE (
    node_id INTEGER,
    node_name VARCHAR(255),
    node_type VARCHAR(50),
    node_file_path TEXT,
    level INTEGER
)
AS $$
BEGIN
    RETURN QUERY
    WITH RECURSIVE node_hierarchy AS (
        -- Anchor member: Select the starting node
        SELECT
            ce.id,
            ce.name,
            ce.type,
            ce.file_path,
            0 AS current_level -- Level relative to the p_start_node_id
        FROM
            code_elements ce
        WHERE
            ce.id = p_start_node_id

        UNION ALL

        -- Recursive member: Select children of nodes from the previous level
        SELECT
            child_ce.id,
            child_ce.name,
            child_ce.type,
            child_ce.file_path,
            nh.current_level + 1
        FROM
            code_elements child_ce
        JOIN
            dependencies d ON child_ce.id = d.child_id
        JOIN
            node_hierarchy nh ON d.parent_id = nh.id
        -- WHERE nh.current_level < 20 -- Optional: safety limit for recursion depth
                                    -- for very deep or potentially cyclic (if data error) graphs
    )
    SELECT
        nh.id,
        nh.name,
        nh.type,
        nh.file_path,
        nh.current_level
    FROM
        node_hierarchy nh;
END;
$$ LANGUAGE plpgsql;

-- Assuming you want to find all descendants of the code_element with id = 5
SELECT *
FROM get_node_descendants_with_levels(5)
ORDER BY level, node_name;

-- If you know the name, type, and path of the starting node, you can find its ID first:
SELECT *
FROM get_node_descendants_with_levels(
    (SELECT id FROM code_elements WHERE name = 'MyMainClass' AND type = 'class' AND file_path = 'src/main.py')
)
ORDER BY level, node_name;

-- If you only want children and not the starting node itself:
SELECT *
FROM get_node_descendants_with_levels(5)
WHERE level > 0
ORDER BY level, node_name;



--A General VIEW (Less Ideal for this Specific Parameterized Use Case)
CREATE VIEW all_descendant_paths_with_levels AS
WITH RECURSIVE node_hierarchy_all AS (
    SELECT
        ce.id as originating_node_id, -- The ultimate ancestor for this path
        ce.id as current_node_id,
        ce.name as current_node_name,
        ce.type as current_node_type,
        ce.file_path as current_node_file_path,
        0 AS level_from_originating_node
    FROM
        code_elements ce

    UNION ALL

    SELECT
        nha.originating_node_id,
        child_ce.id,
        child_ce.name,
        child_ce.type,
        child_ce.file_path,
        nha.level_from_originating_node + 1
    FROM
        code_elements child_ce
    JOIN
        dependencies d ON child_ce.id = d.child_id
    JOIN
        node_hierarchy_all nha ON d.parent_id = nha.current_node_id
)
SELECT * FROM node_hierarchy_all;

--USAGE
SELECT
    current_node_id,
    current_node_name,
    current_node_type,
    current_node_file_path,
    level_from_originating_node
FROM
    all_descendant_paths_with_levels
WHERE
    originating_node_id = 5 -- Filter for your desired starting node
ORDER BY
    level_from_originating_node, current_node_name;

