-- Show directories and files under the root after one revision

select revision.id as revision_id, committer_date,  directory as directory_id, dir_entries, file_entries, rev_entries
from revision, directory
where revision.directory = directory.id
and revision.id = '\xb086670d6a0f2a30b562fe4142e0cb82aec0d68e';

-- Show directory structure given a directory_entry_id

select directory_entry_dir.id as directory_entry_dir_entry_id, directory.id as directory_id,
dir_entries, file_entries, rev_entries
from  directory, directory_entry_dir
where directory.id = directory_entry_dir.target
and directory_entry_dir.id = '82275439';


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
