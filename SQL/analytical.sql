
-- BFS search

WITH RECURSIVE search_tree (id, parent_id, parent_rank, depth) AS (
 select id, parent_id, parent_rank, 0
 from revision_history rh
 where rh.parent_id = '\x8be4d8a73b2d327687c053eeccd04551db5d9818'
 UNION ALL
 select rh.id, rh.parent_id, rh.parent_rank, depth + 1
 from revision_history rh, search_tree st
 where st.id = rh.parent_id and depth <= 10
)
select * from search_tree ORDER BY depth;
